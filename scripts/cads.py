import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
import numpy as np
from collections import OrderedDict
from typing import Union

from modules import script_callbacks
from modules.script_callbacks import CFGDenoiserParams
try:
        from modules.rng import randn_like
except ImportError:
        from torch import randn_like

import torch

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))

"""

An implementation of CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling for Automatic1111 Webui

@inproceedings{
    sadat2024cads,
    title={{CADS}: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling},
    author={Seyedmorteza Sadat and Jakob Buhmann and Derek Bradley and Otmar Hilliges and Romann M. Weber},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=zMoNrajk2X}
}

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-cads

"""
class CADSExtensionScript(scripts.Script):
        # Extension title in menu UI
        def title(self):
                return "CADS"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def ui(self, is_img2img):
                with gr.Accordion('CADS', open=False):
                        active = gr.Checkbox(value=False, default=False, label="Active", elem_id='cads_active')
                        rescale = gr.Checkbox(value=True, default=True, label="Rescale CFG", elem_id = 'cads_rescale')
                        with gr.Row():
                                use_step_mode = gr.Checkbox(value=False, default=False, label="Use step sliders", elem_id='cads_use_step_mode', info='Convert Tau sliders to operate on absolute sampler steps instead of percentages.')
                                respect_strength = gr.Checkbox(value=False, default=False, label="Scale steps by strength", elem_id='cads_respect_strength', info='When enabled, CADS step sliders account for denoising strength / t_enc for img2img and hires. fix passes.', interactive=False)
                                step_start = gr.Slider(value = 0, minimum = 0, maximum = 150, step = 1, label="Tau 1 Step", elem_id = 'cads_tau1_step', info="Step on which CADS starts when using step sliders.", interactive=False)
                                step_stop = gr.Slider(value = 0, minimum = 0, maximum = 150, step = 1, label="Tau 2 Step", elem_id = 'cads_tau2_step', info="Step on which CADS stops when using step sliders.", interactive=False)
                        with gr.Row():
                                t1 = gr.Slider(value = 0.6, minimum = 0.0, maximum = 1.0, step = 0.05, label="Tau 1", elem_id = 'cads_tau1', info="Step to start interpolating from full strength. Default 0.6")
                                t2 = gr.Slider(value = 0.9, minimum = 0.0, maximum = 1.0, step = 0.05, label="Tau 2", elem_id = 'cads_tau2', info="Step to stop affecting image. Default 0.9")
                        with gr.Row():
                                noise_scale = gr.Slider(value = 0.25, minimum = 0.0, maximum = 1.0, step = 0.01, label="Noise Scale", elem_id = 'cads_noise_scale', info='Scale of noise injected at every time step, default 0.25, recommended <= 0.3')
                                mixing_factor= gr.Slider(value = 1.0, minimum = 0.0, maximum = 1.0, step = 0.01, label="Mixing Factor", elem_id = 'cads_mixing_factor', info='Regularization factor, lowering this will increase the diversity of the images with more chance of divergence, default 1.0')
                        with gr.Accordion('Experimental', open=False):
                                apply_to_hr_pass = gr.Checkbox(value=False, default=False, label="Apply to Hires. Fix", elem_id='cads_hr_fix_active', info='Requires a very high denoising value to work. Default False')

                def toggle_step_sliders(enabled):
                        slider_state = gr.Slider.update(interactive=enabled)
                        tau_state = gr.Slider.update(interactive=not enabled)
                        checkbox_state = gr.Checkbox.update(interactive=enabled)
                        return slider_state, slider_state, tau_state, tau_state, checkbox_state

                use_step_mode.change(
                        fn=toggle_step_sliders,
                        inputs=use_step_mode,
                        outputs=[step_start, step_stop, t1, t2, respect_strength],
                )

                active.do_not_save_to_config = True
                rescale.do_not_save_to_config = True
                use_step_mode.do_not_save_to_config = True
                respect_strength.do_not_save_to_config = True
                step_start.do_not_save_to_config = True
                step_stop.do_not_save_to_config = True
                t1.do_not_save_to_config = True
                t2.do_not_save_to_config = True
                noise_scale.do_not_save_to_config = True
                mixing_factor.do_not_save_to_config = True
                apply_to_hr_pass.do_not_save_to_config = True
                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='CADS Active' in d)),
                        (rescale, 'CADS Rescale'),
                        (use_step_mode, lambda d: gr.Checkbox.update(value=d.get('CADS Use Step Mode', False))),
                        (respect_strength, lambda d: gr.Checkbox.update(value=d.get('CADS Respect Strength', False))),
                        (step_start, 'CADS Tau 1 Step'),
                        (step_stop, 'CADS Tau 2 Step'),
                        (t1, 'CADS Tau 1'),
                        (t2, 'CADS Tau 2'),
                        (noise_scale, 'CADS Noise Scale'),
                        (mixing_factor, 'CADS Mixing Factor'),
                        (apply_to_hr_pass, 'CADS Apply To Hires. Fix'),
                ]
                self.paste_field_names = [
                        'cads_active',
                        'cads_rescale',
                        'cads_use_step_mode',
                        'cads_respect_strength',
                        'cads_tau1_step',
                        'cads_tau2_step',
                        'cads_tau1',
                        'cads_tau2',
                        'cads_noise_scale',
                        'cads_mixing_factor',
                        'cads_hr_fix_active',
                ]
                return [active, use_step_mode, respect_strength, step_start, step_stop, t1, t2, noise_scale, mixing_factor, rescale, apply_to_hr_pass]

        def before_process_batch(self, p, active, use_step_mode, respect_strength, step_start, step_stop, t1, t2, noise_scale, mixing_factor, rescale, apply_to_hr_pass, *args, **kwargs):
                self.unhook_callbacks()
                active = getattr(p, "cads_active", active)
                if active is False:
                        return
                use_step_mode = getattr(p, "cads_use_step_mode", use_step_mode)
                respect_strength = getattr(p, "cads_respect_strength", respect_strength)
                step_start = getattr(p, "cads_tau1_step", step_start)
                step_stop = getattr(p, "cads_tau2_step", step_stop)
                steps = getattr(p, "steps", -1)
                if use_step_mode and steps <= 0:
                        logger.error("Steps not set, disabling CADS step sliders")
                        use_step_mode = False

                strength_scale = 1.0
                effective_steps_float = float(max(steps, 1))
                effective_step_count = max(int(round(effective_steps_float)), 1)
                if use_step_mode:
                        step_start = int(max(min(step_start, steps), 0))
                        step_stop = int(max(min(step_stop, steps), 0))
                        if step_stop < step_start:
                                step_stop = step_start
                        steps_float = float(max(steps, 1))
                        if respect_strength:
                                strength_raw = getattr(p, "denoising_strength", None)
                                try:
                                        strength_scale = float(strength_raw)
                                except (TypeError, ValueError):
                                        strength_scale = 1.0
                                if not np.isfinite(strength_scale):
                                        strength_scale = 1.0
                                strength_scale = max(min(strength_scale, 1.0), 0.0)
                                if strength_scale == 0.0:
                                        logger.warning("CADS: Denoising strength is 0.0, falling back to 1.0 for step conversion")
                                        strength_scale = 1.0
                                effective_steps_float = max(steps_float * strength_scale, 1.0)
                                effective_step_count = max(int(round(effective_steps_float)), 1)
                                step_start = int(max(min(step_start, effective_step_count), 0))
                                step_stop = int(max(min(step_stop, effective_step_count), 0))
                                if step_stop < step_start:
                                        step_stop = step_start
                                steps_float = effective_steps_float
                        t1 = max(min(1.0 - (step_stop / steps_float), 1.0), 0.0)
                        t2 = max(min(1.0 - (step_start / steps_float), 1.0), 0.0)
                else:
                        step_start = 0
                        step_stop = 0
                        t1 = getattr(p, "cads_tau1", t1)
                        t2 = getattr(p, "cads_tau2", t2)
                        strength_scale = 1.0
                        effective_step_count = max(int(round(float(max(steps, 1)))), 1)
                noise_scale = getattr(p, "cads_noise_scale", noise_scale)
                mixing_factor = getattr(p, "cads_mixing_factor", mixing_factor)
                rescale = getattr(p, "cads_rescale", rescale)
                apply_to_hr_pass = getattr(p, "cads_hr_fix_active", apply_to_hr_pass)

                first_pass_steps = getattr(p, "steps", -1)
                if first_pass_steps <= 0:
                        logger.error("Steps not set, disabling CADS")
                        return

                setattr(p, "cads_use_step_mode", use_step_mode)
                setattr(p, "cads_respect_strength", respect_strength)
                setattr(p, "cads_tau1_step", step_start)
                setattr(p, "cads_tau2_step", step_stop)
                setattr(p, "cads_tau1", t1)
                setattr(p, "cads_tau2", t2)
                setattr(p, "cads_strength_scale", strength_scale)
                setattr(p, "cads_effective_steps", effective_step_count)

                if not hasattr(p, "extra_generation_params") or not isinstance(p.extra_generation_params, dict):
                        p.extra_generation_params = {}

                p.extra_generation_params.update({
                        "CADS Active": active,
                        "CADS Use Step Mode": use_step_mode,
                        "CADS Respect Strength": respect_strength,
                        "CADS Tau 1 Step": step_start,
                        "CADS Tau 2 Step": step_stop,
                        "CADS Tau 1": t1,
                        "CADS Tau 2": t2,
                        "CADS Noise Scale": noise_scale,
                        "CADS Mixing Factor": mixing_factor,
                        "CADS Rescale": rescale,
                        "CADS Apply To Hires. Fix": apply_to_hr_pass,
                        "CADS Strength Scale": strength_scale,
                        "CADS Effective Steps": effective_step_count,
                })
                self.create_hook(p, active, t1, t2, noise_scale, mixing_factor, rescale, effective_step_count)
        
        def create_hook(self, p, active, t1, t2, noise_scale, mixing_factor, rescale, total_sampling_steps, *args, **kwargs):
                # Use lambda to call the callback function with the parameters to avoid global variables
                y = lambda params: self.on_cfg_denoiser_callback(params, t1=t1, t2=t2, noise_scale=noise_scale, mixing_factor=mixing_factor, rescale=rescale, total_sampling_steps=total_sampling_steps)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(y)
                script_callbacks.on_script_unloaded(self.unhook_callbacks)

        def postprocess_batch(self, p, active, use_step_mode, respect_strength, step_start, step_stop, t1, t2, noise_scale, mixing_factor, rescale, apply_to_hr_pass, *args, **kwargs):
                self.unhook_callbacks()

        def unhook_callbacks(self):
                logger.debug('Unhooked callbacks')
                script_callbacks.remove_current_script_callbacks()

        def cads_linear_schedule(self, t, tau1, tau2):
                """ CADS annealing schedule function """
                if t <= tau1:
                        return 1.0
                if t>= tau2:
                        return 0.0
                gamma = (tau2-t)/(tau2-tau1)
                return gamma

        def add_noise(self, y, gamma, noise_scale, psi, rescale=False):
                """ CADS adding noise to the condition

                Arguments:
                y: Input conditioning
                gamma: Noise level w.r.t t
                noise_scale (float): Noise scale
                psi (float): Rescaling factor
                rescale (bool): Rescale the condition
                """
                y_mean, y_std = torch.mean(y), torch.std(y)
                y = np.sqrt(gamma) * y + noise_scale * np.sqrt(1-gamma) * randn_like(y)
                if rescale:
                        y_scaled = (y - torch.mean(y)) / torch.std(y) * y_std + y_mean
                        if not torch.isnan(y_scaled).any():
                                y = psi * y_scaled + (1 - psi) * y
                        else:
                                logger.debug("Warning: NaN encountered in rescaling")
                return y

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, t1, t2, noise_scale, mixing_factor, rescale, total_sampling_steps):
                sampling_step = params.sampling_step
                total_sampling_step = total_sampling_steps
                text_cond = params.text_cond
                text_uncond = params.text_uncond

                t = 1.0 - max(min(sampling_step / total_sampling_step, 1.0), 0.0) # Algorithms assumes we start at 1.0 and go to 0.0
                gamma = self.cads_linear_schedule(t, t1, t2)
                # SD 1.5
                if isinstance(text_cond, torch.Tensor) and isinstance(text_uncond, torch.Tensor):
                        params.text_cond = self.add_noise(text_cond, gamma, noise_scale, mixing_factor, rescale)
                        params.text_uncond = self.add_noise(text_uncond, gamma, noise_scale, mixing_factor, rescale)
                # SDXL
                elif isinstance(text_cond, Union[dict, OrderedDict]) and isinstance(text_uncond, Union[dict, OrderedDict]):
                        params.text_cond['crossattn'] = self.add_noise(text_cond['crossattn'], gamma, noise_scale, mixing_factor, rescale)
                        params.text_uncond['crossattn'] = self.add_noise(text_uncond['crossattn'], gamma, noise_scale, mixing_factor, rescale)
                        params.text_cond['vector'] = self.add_noise(text_cond['vector'], gamma, noise_scale, mixing_factor, rescale)
                        params.text_uncond['vector'] = self.add_noise(text_uncond['vector'], gamma, noise_scale, mixing_factor, rescale)
                else:
                        logger.error('Unknown text_cond type')
                        pass
        
        def before_hr(self, p, *args):
                self.unhook_callbacks()

                params = getattr(p, "extra_generation_params", None)
                if not params:
                        logger.error("Missing attribute extra_generation_params")
                        return

                active = params.get("CADS Active", False)
                if active is False:
                        return

                apply_to_hr_pass = params.get("CADS Apply To Hires. Fix", False)
                if apply_to_hr_pass is False:
                        logger.debug("Disabled for hires. fix")
                        return

                t1 = params.get("CADS Tau 1", None)
                t2 = params.get("CADS Tau 2", None)
                noise_scale = params.get("CADS Noise Scale", None)
                mixing_factor = params.get("CADS Mixing Factor", None)
                rescale = params.get("CADS Rescale", None)
                use_step_mode = params.get("CADS Use Step Mode", False)
                respect_strength = params.get("CADS Respect Strength", False)
                step_start = params.get("CADS Tau 1 Step", 0)
                step_stop = params.get("CADS Tau 2 Step", 0)

                if t1 is None or t2 is None or noise_scale is None or mixing_factor is None or rescale is None:
                        logger.error("Missing needed parameters for Hires. fix")
                        return

                hr_pass_steps = getattr(p, "hr_second_pass_steps", -1)
                if hr_pass_steps < 0:
                        logger.error("Attribute hr_second_pass_steps not found")
                        return
                if hr_pass_steps == 0:
                        logger.debug("Using first pass step count for hires. fix")
                        hr_pass_steps = getattr(p, "steps", -1)

                if use_step_mode and hr_pass_steps > 0:
                        steps_float = float(max(hr_pass_steps, 1))
                        strength_scale = params.get("CADS Strength Scale", 1.0)
                        if respect_strength:
                                strength_raw = getattr(p, "hr_denoising_strength", getattr(p, "denoising_strength", None))
                                try:
                                        strength_scale = float(strength_raw)
                                except (TypeError, ValueError):
                                        strength_scale = 1.0
                                if not np.isfinite(strength_scale):
                                        strength_scale = 1.0
                                strength_scale = max(min(strength_scale, 1.0), 0.0)
                                if strength_scale == 0.0:
                                        logger.warning("CADS: Hires. fix denoising strength is 0.0, falling back to 1.0 for step conversion")
                                        strength_scale = 1.0
                                steps_float = max(steps_float * strength_scale, 1.0)
                        effective_step_count = max(int(round(steps_float)), 1)
                        step_start = int(max(min(step_start, effective_step_count), 0))
                        step_stop = int(max(min(step_stop, effective_step_count), 0))
                        if step_stop < step_start:
                                step_stop = step_start
                        t1 = max(min(1.0 - (step_stop / steps_float), 1.0), 0.0)
                        t2 = max(min(1.0 - (step_start / steps_float), 1.0), 0.0)
                        params["CADS Tau 1 Step"] = step_start
                        params["CADS Tau 2 Step"] = step_stop
                        params["CADS Tau 1"] = t1
                        params["CADS Tau 2"] = t2
                        if respect_strength:
                                params["CADS Strength Scale"] = strength_scale
                                params["CADS Effective Steps"] = effective_step_count

                logger.debug("Enabled for hi-res fix with %i steps, re-hooking CADS", hr_pass_steps)
                total_sampling_steps = hr_pass_steps
                if use_step_mode and respect_strength:
                        total_sampling_steps = params.get("CADS Effective Steps", max(int(round(float(max(hr_pass_steps, 1)))), 1))
                self.create_hook(p, active, t1, t2, noise_scale, mixing_factor, rescale, total_sampling_steps)


# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def cads_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
    return fun

def cads_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "cads_active"):
                setattr(p, "cads_active", True)
        setattr(p, field, x)

    return fun

def cads_apply_step_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "cads_active"):
                setattr(p, "cads_active", True)
        setattr(p, "cads_use_step_mode", True)
        setattr(p, field, x)

    return fun

def make_axis_options():
        xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
        # Add the boolean choice function to SD.Next XYZ Grid script
        if not hasattr(xyz_grid, "boolean_choice"):
                xyz_grid.boolean_choice = lambda reverse=False: ["True", "False"] if not reverse else ["False", "True"]
        extra_axis_options = {
                xyz_grid.AxisOption("[CADS] Active", str, cads_apply_override('cads_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                xyz_grid.AxisOption("[CADS] Rescale CFG", str, cads_apply_override('cads_rescale', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                xyz_grid.AxisOption("[CADS] Use Step Mode", str, cads_apply_override('cads_use_step_mode', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                xyz_grid.AxisOption("[CADS] Respect Strength", str, cads_apply_override('cads_respect_strength', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                xyz_grid.AxisOption("[CADS] Tau 1 Step", int, cads_apply_step_field("cads_tau1_step")),
                xyz_grid.AxisOption("[CADS] Tau 2 Step", int, cads_apply_step_field("cads_tau2_step")),
                xyz_grid.AxisOption("[CADS] Tau 1", float, cads_apply_field("cads_tau1")),
                xyz_grid.AxisOption("[CADS] Tau 2", float, cads_apply_field("cads_tau2")),
                xyz_grid.AxisOption("[CADS] Noise Scale", float, cads_apply_field("cads_noise_scale")),
                xyz_grid.AxisOption("[CADS] Mixing Factor", float, cads_apply_field("cads_mixing_factor")),
                xyz_grid.AxisOption("[CADS] Apply to Hires. Fix", str, cads_apply_override('cads_hr_fix_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
        }
        if not any("[CADS]" in x.label for x in xyz_grid.axis_options):
                xyz_grid.axis_options.extend(extra_axis_options)

def callback_before_ui():
        try:
                make_axis_options()
        except:
                logger.exception("CADS: Error while making axis options")

script_callbacks.on_before_ui(callback_before_ui)
