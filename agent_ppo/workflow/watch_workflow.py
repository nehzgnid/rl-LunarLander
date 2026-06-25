from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from agent_ppo.conf.conf import Config
from agent_ppo.workflow.env_factory import BASE_ENV_MODE, ENV_MODES, REAL_ENV_MODE
from agent_ppo.workflow.evaluate_workflow import _make_env, _make_normalizer, _model_path, _resolve_exp_id, _vecnormalize_path


PANEL_WIDTH = 360
WINDOW_PADDING = 18
BUTTON_H = 36
BUTTON_GAP = 10
MAX_VIEW_W = 960
MAX_VIEW_H = 640
MIN_VIEW_W = 360
MIN_VIEW_H = 240
FPS = 30
STEP_INTERVAL_MS = 120

BG = (232, 234, 238)
PANEL_BG = (246, 247, 249)
PANEL_BORDER = (205, 210, 218)
TEXT = (30, 34, 42)
TEXT_MUTED = (92, 99, 112)
BUTTON_BG = (230, 234, 242)
BUTTON_ACTIVE = (214, 225, 255)
BUTTON_TEXT = (25, 29, 36)
GOOD = (31, 150, 74)
BAD = (190, 55, 60)


class Button:
    def __init__(self, rect, label: str):
        self.rect = rect
        self.label = label

    def draw(self, surface, font, active: bool = False):
        import pygame

        bg = BUTTON_ACTIVE if active else BUTTON_BG
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, 1, border_radius=6)
        text = font.render(self.label, True, BUTTON_TEXT)
        surface.blit(text, text.get_rect(center=self.rect.center))

    def hit(self, pos) -> bool:
        return self.rect.collidepoint(pos)


class GymViewer:
    def __init__(
        self,
        exp_id: int | None,
        n_timesteps: int | None = None,
        device: str = "auto",
        env_mode: str = BASE_ENV_MODE,
    ):
        import pygame

        if env_mode not in ENV_MODES:
            raise ValueError(f"Unsupported env_mode: {env_mode}. Expected one of {ENV_MODES}")

        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("microsoftyahei,simhei,arial", 18)
        self.small_font = pygame.font.SysFont("microsoftyahei,simhei,arial", 14)
        self.title_font = pygame.font.SysFont("microsoftyahei,simhei,arial", 22, bold=True)

        self.env_mode = env_mode
        self.exp_id = _resolve_exp_id(exp_id, env_mode=self.env_mode)
        self.device = device
        self.max_steps = int(n_timesteps or 1000)
        self.env = _make_env(render_mode="rgb_array", env_mode=self.env_mode)
        self.model_path = None
        self.normalizer = None
        self.model = None
        self._load_exp(self.exp_id)

        self.obs = None
        self.done = False
        self.auto_run = False
        self.step_count = 0
        self.episode = 1
        self.episode_reward = 0.0
        self.last_reward = 0.0
        self.status = "就绪"
        self.last_step_time = 0

        self.frame = self._render_frame_after_reset()
        self.width, self.height = self._initial_window_size()
        self.view_w = 1
        self.view_h = 1
        self._update_view_size(self.width, self.height)

        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption(f"LunarLander PPO Viewer ({self.env_mode})")
        self.clock = pygame.time.Clock()
        self._build_layout()

    def _initial_window_size(self):
        import pygame

        frame_h, frame_w = self.frame.shape[:2]
        display_info = pygame.display.Info()
        max_screen_w = max(display_info.current_w - 40, PANEL_WIDTH + MIN_VIEW_W + WINDOW_PADDING * 3)
        max_screen_h = max(display_info.current_h - 80, MIN_VIEW_H + WINDOW_PADDING * 2)

        max_view_w = min(MAX_VIEW_W, max_screen_w - PANEL_WIDTH - WINDOW_PADDING * 3)
        max_view_h = min(MAX_VIEW_H, max_screen_h - WINDOW_PADDING * 2)
        scale = min(max_view_w / max(frame_w, 1), max_view_h / max(frame_h, 1))
        view_w = max(MIN_VIEW_W, int(frame_w * scale))
        view_h = max(MIN_VIEW_H, int(frame_h * scale))

        width = min(max_screen_w, PANEL_WIDTH + view_w + WINDOW_PADDING * 3)
        height = min(max_screen_h, max(view_h + WINDOW_PADDING * 2, 590))
        return int(width), int(height)

    def _update_view_size(self, width: int, height: int):
        frame_h, frame_w = self.frame.shape[:2]
        max_view_w = max(width - PANEL_WIDTH - WINDOW_PADDING * 3, 1)
        max_view_h = max(height - WINDOW_PADDING * 2, 1)
        scale = min(max_view_w / max(frame_w, 1), max_view_h / max(frame_h, 1))
        self.view_w = max(1, int(frame_w * scale))
        self.view_h = max(1, int(frame_h * scale))

    def _build_layout(self):
        import pygame

        x = WINDOW_PADDING
        panel_w = PANEL_WIDTH - WINDOW_PADDING * 2
        half = (panel_w - BUTTON_GAP) // 2
        self.buttons = {
            "base": Button(pygame.Rect(x, 126, half, BUTTON_H), "Base"),
            "real": Button(pygame.Rect(x + half + BUTTON_GAP, 126, half, BUTTON_H), "Real"),
            "run": Button(pygame.Rect(x, 176, half, BUTTON_H), "开始"),
            "pause": Button(pygame.Rect(x + half + BUTTON_GAP, 176, half, BUTTON_H), "暂停"),
            "step": Button(pygame.Rect(x, 222, half, BUTTON_H), "单步"),
            "reset": Button(pygame.Rect(x + half + BUTTON_GAP, 222, half, BUTTON_H), "重置"),
            "choose_model": Button(pygame.Rect(x, 268, panel_w, BUTTON_H), "选择模型文件"),
            "quit": Button(pygame.Rect(x, 314, panel_w, BUTTON_H), "退出"),
        }

    def _load_exp(self, exp_id: int):
        self.exp_id = int(exp_id)
        self.model_path = _model_path(self.exp_id, env_mode=self.env_mode)
        self.normalizer = _make_normalizer(_vecnormalize_path(self.exp_id, env_mode=self.env_mode))
        self.model = _load_model(self.model_path, device=self.device)
        self.status = f"已加载 {self.env_mode} 实验 {self.exp_id}"

    def _load_latest_for_current_mode(self):
        self.exp_id = _resolve_exp_id(None, env_mode=self.env_mode)
        self._load_exp(self.exp_id)

    def _switch_env_mode(self, env_mode: str):
        import pygame

        if env_mode == self.env_mode:
            return
        if env_mode not in ENV_MODES:
            self.status = f"不支持的环境模式: {env_mode}"
            return

        old_env = self.env
        try:
            old_env.close()
        except Exception:
            pass

        previous_mode = self.env_mode
        try:
            self.env_mode = env_mode
            self.env = _make_env(render_mode="rgb_array", env_mode=self.env_mode)
            self._load_latest_for_current_mode()
            self.done = False
            self.auto_run = False
            self.step_count = 0
            self.episode_reward = 0.0
            self.last_reward = 0.0
            self.frame = self._render_frame_after_reset()
            self._update_view_size(self.width, self.height)
            pygame.display.set_caption(f"LunarLander PPO Viewer ({self.env_mode})")
            self.status = f"已切换到 {self.env_mode}"
        except Exception as exc:
            self.env_mode = previous_mode
            self.env = _make_env(render_mode="rgb_array", env_mode=self.env_mode)
            self._load_latest_for_current_mode()
            self.frame = self._render_frame_after_reset()
            self.status = f"切换失败: {exc}"

    def _load_model_file(self, model_path: Path):
        self.model_path = model_path
        self.normalizer = _make_normalizer(_normalizer_for_model_path(model_path))
        self.model = _load_model(model_path, device=self.device)
        self.status = f"已加载 {model_path.name}"
        self.reset_episode()

    def _render_frame_after_reset(self):
        reset_out = self.env.reset()
        self.obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        frame = self.env.render()
        return _ensure_rgb_frame(frame)

    def reset_episode(self):
        self.done = False
        self.auto_run = False
        self.step_count = 0
        self.episode_reward = 0.0
        self.last_reward = 0.0
        self.status = "已重置"
        self.frame = self._render_frame_after_reset()

    def step_agent(self):
        if self.done:
            return
        if self.step_count >= self.max_steps:
            self.done = True
            self.auto_run = False
            self.status = "达到最大步数"
            return

        norm_obs = self.normalizer.normalize(self.obs)[None, :]
        action, _ = self.model.predict(norm_obs, deterministic=True)
        action = int(np.asarray(action).reshape(-1)[0])
        self.obs, reward, terminated, truncated, _info = self.env.step(action)

        self.last_reward = float(reward)
        self.episode_reward += self.last_reward
        self.step_count += 1
        self.frame = _ensure_rgb_frame(self.env.render())
        self.status = f"动作 {action}"
        if terminated or truncated:
            self.done = True
            self.auto_run = False
            self.status = "回合结束"

    def _draw(self):
        self.screen.fill(BG)
        self._draw_environment()
        self._draw_panel()
        import pygame

        pygame.display.flip()

    def _draw_environment(self):
        import pygame

        view_left = PANEL_WIDTH + WINDOW_PADDING * 2
        view_top = WINDOW_PADDING + max(0, (self.height - WINDOW_PADDING * 2 - self.view_h) // 2)
        surf = pygame.surfarray.make_surface(np.transpose(self.frame, (1, 0, 2)))
        surf = pygame.transform.smoothscale(surf, (self.view_w, self.view_h))
        self.screen.blit(surf, (view_left, view_top))
        pygame.draw.rect(self.screen, PANEL_BORDER, (view_left, view_top, self.view_w, self.view_h), 1)

    def _draw_panel(self):
        import pygame

        panel = pygame.Rect(WINDOW_PADDING, WINDOW_PADDING, PANEL_WIDTH - WINDOW_PADDING * 2, self.height - WINDOW_PADDING * 2)
        pygame.draw.rect(self.screen, PANEL_BG, panel)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel, 1)

        x = WINDOW_PADDING + 16
        y = WINDOW_PADDING + 18
        self.screen.blit(self.title_font.render("LunarLander PPO", True, TEXT), (x, y))
        y += 36
        for line in [
            f"环境: {Config.ENV_ID}",
            f"模式: {self.env_mode}",
            f"实验: {self.exp_id}",
        ]:
            self.screen.blit(self.small_font.render(line, True, TEXT_MUTED), (x, y))
            y += 22

        self.buttons["base"].draw(self.screen, self.font, active=self.env_mode == BASE_ENV_MODE)
        self.buttons["real"].draw(self.screen, self.font, active=self.env_mode == REAL_ENV_MODE)
        self.buttons["run"].draw(self.screen, self.font, active=self.auto_run)
        self.buttons["pause"].draw(self.screen, self.font, active=not self.auto_run)
        self.buttons["step"].draw(self.screen, self.font)
        self.buttons["reset"].draw(self.screen, self.font)
        self.buttons["choose_model"].draw(self.screen, self.font)
        self.buttons["quit"].draw(self.screen, self.font)

        y = 365
        status_color = GOOD if self.episode_reward >= Config.EVAL_SUCCESS_REWARD else (BAD if self.done else TEXT)
        lines = [
            f"模型: {self.model_path.name}",
            f"画面: {self.view_w}x{self.view_h}",
            f"状态: {self.status}",
            f"回合: {self.episode}",
            f"步数: {self.step_count}/{self.max_steps}",
            f"本步奖励: {self.last_reward:.3f}",
            f"累计奖励: {self.episode_reward:.3f}",
            f"是否结束: {self.done}",
            "",
            "快捷键",
            "Space 开始/暂停",
            "S 单步",
            "R 重置",
            "M 选择模型文件",
            "Esc 退出",
        ]
        for line in lines:
            color = status_color if line.startswith("状态:") else TEXT
            self.screen.blit(self.small_font.render(line, True, color), (x, y))
            y += 22

    def _handle_click(self, pos):
        if self.buttons["base"].hit(pos):
            self._switch_env_mode(BASE_ENV_MODE)
        elif self.buttons["real"].hit(pos):
            self._switch_env_mode(REAL_ENV_MODE)
        elif self.buttons["run"].hit(pos):
            self.auto_run = True
            self.status = "运行中"
        elif self.buttons["pause"].hit(pos):
            self.auto_run = False
            self.status = "已暂停"
        elif self.buttons["step"].hit(pos):
            self.step_agent()
        elif self.buttons["reset"].hit(pos):
            self.reset_episode()
        elif self.buttons["choose_model"].hit(pos):
            self._choose_model_file()
        elif self.buttons["quit"].hit(pos):
            return False
        return True

    def _handle_key(self, key):
        import pygame

        if key == pygame.K_SPACE:
            self.auto_run = not self.auto_run
            self.status = "运行中" if self.auto_run else "已暂停"
        elif key == pygame.K_s:
            self.step_agent()
        elif key == pygame.K_r:
            self.reset_episode()
        elif key == pygame.K_m:
            self._choose_model_file()
        elif key == pygame.K_ESCAPE:
            return False
        return True

    def _choose_model_file(self):
        selected = _open_model_file_dialog()
        if selected is None:
            self.status = "已取消选择"
            return
        try:
            self._load_model_file(selected)
        except Exception as exc:
            self.status = f"加载失败: {exc}"

    def run(self):
        import pygame

        running = True
        while running:
            now = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    running = self._handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
                elif event.type == pygame.VIDEORESIZE:
                    self.width = max(event.w, PANEL_WIDTH + MIN_VIEW_W + WINDOW_PADDING * 3)
                    self.height = max(event.h, 560)
                    self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                    self._update_view_size(self.width, self.height)

            if self.auto_run and now - self.last_step_time >= STEP_INTERVAL_MS:
                self.step_agent()
                self.last_step_time = now

            self._draw()
            self.clock.tick(FPS)

        self.env.close()
        pygame.quit()


def _load_model(model_path: Path, device: str = "auto"):
    return PPO.load(
        str(model_path),
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
        device=device,
    )


def _normalizer_for_model_path(model_path: Path) -> Path:
    run_dir = model_path.parent
    stats_dir = run_dir / Config.ENV_ID
    if model_path.name == "best_model.zip":
        best_stats = stats_dir / "best_vecnormalize.pkl"
        if best_stats.exists():
            return best_stats
    return stats_dir / "vecnormalize.pkl"


def _open_model_file_dialog() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        initial_dir = Config.LOG_FOLDER / Config.ALGO
        filename = filedialog.askopenfilename(
            title="选择 PPO 模型 zip 文件",
            initialdir=str(initial_dir if initial_dir.exists() else Config.ROOT_DIR),
            filetypes=[("Stable-Baselines3 model", "*.zip"), ("All files", "*.*")],
        )
        root.destroy()
        if not filename:
            return None
        return Path(filename)
    except Exception:
        return None


def _ensure_rgb_frame(frame):
    if frame is None:
        raise RuntimeError("Environment did not return an rgb_array frame. Create it with render_mode='rgb_array'.")
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"Unsupported render frame shape: {frame.shape}")
    return frame.astype(np.uint8)


def _run_no_render(exp_id: int | None, total_steps: int, device: str = "auto", env_mode: str = BASE_ENV_MODE) -> None:
    exp_id = _resolve_exp_id(exp_id, env_mode=env_mode)
    model_path = _model_path(exp_id, env_mode=env_mode)
    normalizer = _make_normalizer(_vecnormalize_path(exp_id, env_mode=env_mode))
    model = _load_model(model_path, device=device)
    env = _make_env(env_mode=env_mode)
    try:
        obs, _info = env.reset()
        episode_reward = 0.0
        episode_len = 0
        episode = 1

        print(f"model: {model_path}")
        print(f"env_mode: {env_mode}")
        print(f"timesteps: {total_steps}")

        for _ in range(total_steps):
            norm_obs = normalizer.normalize(obs)[None, :]
            action, _ = model.predict(norm_obs, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
            obs, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

            episode_reward += float(reward)
            episode_len += 1
            if done:
                print(f"episode {episode}: reward={episode_reward:.3f}, length={episode_len}")
                obs, _info = env.reset()
                episode_reward = 0.0
                episode_len = 0
                episode += 1
    finally:
        env.close()


def workflow(
    no_render: bool = False,
    n_timesteps: int | None = None,
    exp_id: int | None = None,
    device: str = "auto",
    env_mode: str = BASE_ENV_MODE,
) -> None:
    if env_mode not in ENV_MODES:
        raise ValueError(f"Unsupported env_mode: {env_mode}. Expected one of {ENV_MODES}")
    total_steps = int(n_timesteps or 1000)
    run_id = _resolve_exp_id(exp_id, env_mode=env_mode)
    if no_render:
        _run_no_render(run_id, total_steps, device=device, env_mode=env_mode)
        return
    GymViewer(run_id, n_timesteps=total_steps, device=device, env_mode=env_mode).run()
