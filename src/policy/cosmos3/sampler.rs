//! Cosmos3 Edge 的 uniPC 采样器。
//!
//! 与 `diffusers` 的 `UniPCMultistepScheduler` 对齐：`flow_prediction`、`use_flow_sigmas`、
//! `solver_order=2`、`solver_type=bh2`、`predict_x0`、`lower_order_final`。
//! 视频与动作各走一条轨迹，逐元素更新，等价于把两者拼在一起求解。

// 采样器在整条流水线接通前不参与运行时构建；届时移除此豁免。
#![allow(dead_code)]

use burn::tensor::Tensor;

/// sigma 序列：`steps` 个噪声水平，末尾补 0。
#[derive(Clone, Debug)]
pub(super) struct Schedule {
    pub(super) sigmas: Vec<f32>,
    pub(super) num_train_timesteps: usize,
}

impl Schedule {
    /// 平移到 `shift * s / (1 + (shift - 1) * s)`，先按 float64 计算再落成 float32。
    pub(super) fn new(steps: usize, shift: f64, num_train_timesteps: usize) -> Self {
        let sigma_max = (1.0 - 1.0 / num_train_timesteps as f64) as f32 as f64;
        let mut sigmas = Vec::with_capacity(steps);
        for index in 0..steps {
            let raw = sigma_max * (1.0 - index as f64 / steps as f64);
            let shifted = shift * raw / (1.0 + (shift - 1.0) * raw);
            sigmas.push(shifted as f32);
        }
        // 官方为避免首步 `log(alpha)` 出现无穷大会把 1 稍微下移。
        if (sigmas[0] - 1.0).abs() < 1e-6 {
            sigmas[0] -= 1e-6;
        }
        sigmas.push(0.0);
        Self {
            sigmas,
            num_train_timesteps,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.sigmas.len() - 1
    }

    /// 传给模型的时间步，与官方 `sigmas * num_train_timesteps` 的取整一致。
    pub(super) fn timestep(&self, index: usize) -> i64 {
        (self.sigmas[index] * self.num_train_timesteps as f32) as i64
    }

    fn alpha_sigma(sigma: f32) -> (f32, f32) {
        (1.0 - sigma, sigma)
    }

    fn lambda(sigma: f32) -> f32 {
        let (alpha, sigma) = Self::alpha_sigma(sigma);
        alpha.ln() - sigma.ln()
    }
}

/// 一条轨迹的采样状态。
pub(super) struct Trajectory<const D: usize> {
    schedule: Schedule,
    step: usize,
    lower_order: usize,
    /// `[上一步, 当前]` 的转换后模型输出。
    model_outputs: [Option<Tensor<D>>; 2],
    last_sample: Option<Tensor<D>>,
}

impl<const D: usize> Trajectory<D> {
    pub(super) fn new(schedule: Schedule) -> Self {
        Self {
            schedule,
            step: 0,
            lower_order: 0,
            model_outputs: [None, None],
            last_sample: None,
        }
    }

    pub(super) fn timestep(&self) -> i64 {
        self.schedule.timestep(self.step)
    }

    /// 推进一步：输入当前样本与模型给出的速度，返回下一个样本。
    pub(super) fn step(&mut self, model_output: Tensor<D>, sample: Tensor<D>) -> Tensor<D> {
        let index = self.step;
        let sigma = self.schedule.sigmas[index];
        // `flow_prediction`：x0 = x - sigma * v。
        let converted = sample.clone() - model_output.clone() * sigma as f64;

        let corrected = match (&self.last_sample, index > 0) {
            (Some(last), true) => {
                let order = self.lower_order.min(2);
                self.correct(&converted, last, order)
            }
            _ => sample.clone(),
        };

        self.model_outputs[0] = self.model_outputs[1].take();
        self.model_outputs[1] = Some(converted);

        let remaining = self.schedule.len() - index;
        let this_order = remaining.min(2).min(self.lower_order + 1);
        let previous = self.predict(index, &corrected, this_order);
        self.lower_order = (self.lower_order + 1).min(2);
        self.last_sample = Some(corrected);
        self.step += 1;
        previous
    }

    /// 修正步：用上一步的样本重估当前样本。
    fn correct(
        &self,
        model_output: &Tensor<D>,
        last_sample: &Tensor<D>,
        order: usize,
    ) -> Tensor<D> {
        let index = self.step;
        let (alpha_t, sigma_t) = Schedule::alpha_sigma(self.schedule.sigmas[index]);
        let (_, sigma_s0) = Schedule::alpha_sigma(self.schedule.sigmas[index - 1]);
        let h = Schedule::lambda(self.schedule.sigmas[index])
            - Schedule::lambda(self.schedule.sigmas[index - 1]);
        let hh = -h;
        let h_phi_1 = hh.exp() - 1.0;
        let b_h = h_phi_1;

        let m0 = self.model_outputs[1]
            .clone()
            .expect("previous model output");
        let mut d1s = Vec::new();
        let mut rks = Vec::new();
        for offset in 1..order {
            let si = index - (offset + 1);
            let rk = (Schedule::lambda(self.schedule.sigmas[si])
                - Schedule::lambda(self.schedule.sigmas[index - 1]))
                / h;
            rks.push(rk);
            let mi = self.model_outputs[0].clone().expect("older model output");
            d1s.push((mi - m0.clone()) / rk as f64);
        }

        let mut h_phi_k = h_phi_1 / hh - 1.0;
        let mut factorial = 1.0f32;
        let mut coefficients = Vec::new();
        for _ in 1..=order {
            coefficients.push(h_phi_k * factorial / b_h);
            factorial *= 2.0;
            h_phi_k = h_phi_k / hh - 1.0 / factorial;
        }
        let (rho_c, rho_last) = if order == 1 {
            (Vec::new(), 0.5)
        } else {
            // R = [[1, 1], [rks0, 1]]，解出两个系数。
            let r = rks[0];
            let determinant = 1.0 - r;
            let rho0 = (coefficients[0] - coefficients[1]) / determinant;
            let rho1 = (coefficients[1] - r * coefficients[0]) / determinant;
            (vec![rho0], rho1)
        };

        let mut delta = (model_output.clone() - m0.clone()) * rho_last as f64;
        for (rho, d1) in rho_c.iter().zip(d1s.iter()) {
            delta = delta + d1.clone() * *rho as f64;
        }
        last_sample.clone() * (sigma_t / sigma_s0) as f64
            - m0 * (alpha_t * h_phi_1) as f64
            - delta * (alpha_t * b_h) as f64
    }

    /// 预测步：从当前样本推到下一个 sigma。
    fn predict(&self, index: usize, sample: &Tensor<D>, order: usize) -> Tensor<D> {
        let (alpha_t, sigma_t) = Schedule::alpha_sigma(self.schedule.sigmas[index + 1]);
        let (_, sigma_s0) = Schedule::alpha_sigma(self.schedule.sigmas[index]);
        let h = Schedule::lambda(self.schedule.sigmas[index + 1])
            - Schedule::lambda(self.schedule.sigmas[index]);
        let hh = -h;
        let h_phi_1 = hh.exp() - 1.0;
        let b_h = h_phi_1;

        let m0 = self.model_outputs[1].clone().expect("current model output");
        let mut d1s = Vec::new();
        for offset in 1..order {
            let si = index - offset;
            let rk = (Schedule::lambda(self.schedule.sigmas[si])
                - Schedule::lambda(self.schedule.sigmas[index]))
                / h;
            let mi = self.model_outputs[0]
                .clone()
                .expect("previous model output");
            d1s.push((mi - m0.clone()) / rk as f64);
        }

        let mut result =
            sample.clone() * (sigma_t / sigma_s0) as f64 - m0 * (alpha_t * h_phi_1) as f64;
        if let Some(first) = d1s.first() {
            // order 2 时 `rhos_p` 固定为 0.5。
            result = result - first.clone() * (alpha_t * b_h * 0.5) as f64;
        }
        result
    }
}

#[cfg(test)]
mod reference {
    use super::*;
    use burn::tensor::{Device, TensorData};
    use std::path::{Path, PathBuf};

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_le_bytes(*value))
            .collect()
    }

    #[test]
    #[ignore = "需要 cosmos3 采样参照"]
    fn sampler_matches_reference() {
        let parity =
            PathBuf::from(std::env::var("PNEUMA_COSMOS3_SAMPLER").expect("PNEUMA_COSMOS3_SAMPLER"));
        let expected_sigmas = read_f32(&parity.join("sigmas.bin"));
        let expected_steps = read_f32(&parity.join("timesteps.bin"));
        let expected = read_f32(&parity.join("trajectory.bin"));

        let schedule = Schedule::new(4, 5.0, 1000);
        assert_eq!(schedule.sigmas.len(), expected_sigmas.len());
        for (got, want) in schedule.sigmas.iter().zip(&expected_sigmas) {
            assert!((got - want).abs() < 1e-7, "sigma {got} != {want}");
        }
        let mut trajectory = Trajectory::<3>::new(schedule);
        let device = Device::flex();
        let mut sample = Tensor::<3>::from_data(
            TensorData::new(vec![0.5f32, -0.25, 0.75], [1, 1, 3]),
            &device,
        );
        for index in 0..expected_steps.len() {
            let timestep = trajectory.timestep();
            assert_eq!(timestep, expected_steps[index] as i64, "timestep {index}");
            let velocity = sample.clone() * 0.5 + (timestep as f64 / 1000.0).sin() * 0.1;
            sample = trajectory.step(velocity, sample);
            let got: Vec<f32> = sample
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .expect("sample");
            let want = &expected[(index + 1) * 3..(index + 2) * 3];
            let max = got
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("step {index}: max {max:.3e}");
            assert!(max < 2e-5, "step {index} differs by {max:.3e}");
        }
    }
}
