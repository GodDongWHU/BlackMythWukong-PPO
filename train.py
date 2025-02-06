import torch
import torch.nn as nn
from stable_baselines3 import PPO, A2C
import os
from WukongEnv import WukongEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.envs import DummyVecEnv
from ultralytics import YOLO
from torchvision.models import resnet50
import pydirectinput
import time
from walkToBoss import walkToBoss


class YOLOFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(YOLOFeatureExtractor, self).__init__(observation_space, features_dim)
        # 加载预训练的 ResNet50 模型
        self.resnet50 = resnet50(pretrained=True)
        # self.yolo = YOLO("yolov5m.pt")
        # 冻结 ResNet50 所有参数
        for param in self.resnet50.parameters():
            param.requires_grad = False
        # for param in self.yolo.model.parameters():
        #     param.requires_grad = False
        # 去掉 ResNet50 的分类层，只保留特征提取部分
        self.backbone = nn.Sequential(*list(self.resnet50.children())[:-1])
        # self.backbone = self.yolo.model.model[:-1]
        # 定义一个线性层将提取的特征映射到所需维度
        self.fc = nn.Linear(2048, features_dim)  # 2048 是 ResNet50 输出的特征维度
        # self.fc = torch.nn.Linear(640, features_dim)

    def forward(self, observations):
        x = observations["img"].float() / 255.0  # 归一化到 [0, 1]
        x = self.backbone(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class DetailedLossCallback(BaseCallback):
	def __init__(self, save_path, pre_steps, verbose=0):
		super(DetailedLossCallback, self).__init__(verbose)
		self.policy_losses = []
		self.value_losses = []
		self.save_freq = 100  # Save frequency
		self.save_path = save_path
		self.pre_steps = pre_steps

	def _on_step(self) -> bool:
		if self.n_calls % 100 == 0:  # Print every 100 steps
			if hasattr(self.model, 'logger'):
				logs = self.model.logger.name_to_value
				if 'loss/policy_loss' in logs and 'loss/value_loss' in logs:
					policy_loss = logs['loss/policy_loss']
					value_loss = logs['loss/value_loss']
					print(f"Step: {self.n_calls}")
					print(f"  Policy Loss: {policy_loss}")
					print(f"  Value Loss: {value_loss}")
					self.policy_losses.append(policy_loss)
					self.value_losses.append(value_loss)
		if self.n_calls % self.save_freq == 0:
			os.makedirs(self.save_path, exist_ok=True)
			self.model.save(os.path.join(self.save_path, f"PPO_huxianfeng_{self.n_calls + self.pre_steps}"))
			print(f"Model saved at step {self.n_calls}")
		return True

	def on_training_end(self) -> None:
		import matplotlib.pyplot as plt
		plt.figure(figsize=(10, 5))
		plt.plot(self.policy_losses, label='Policy Loss')
		plt.plot(self.value_losses, label='Value Loss')
		plt.legend()
		plt.title('Policy and Value Losses')
		plt.xlabel('Updates (x100 steps)')
		plt.ylabel('Loss')
		plt.savefig('loss_plot.png')
		plt.close()


def train(CREATE_NEW_MODEL, config):
	print("🧠 Training will start soon. This can take a while to initialize...")


	TIMESTEPS = 1			#Learning rate multiplier.
	HORIZON_WINDOW = 500	#Lerning rate number of steps before updating the model. ~2min


	'''Creating folder structure'''
	model_name = "PPO-1"#Your name here
	if not os.path.exists(f"models/{model_name}/"):
		os.makedirs(f"models/{model_name}/")
	if not os.path.exists(f"logs/{model_name}/"):
		os.makedirs(f"logs/{model_name}/")
	models_dir = f"models/{model_name}/"
	logdir = f"logs/{model_name}/"
	model_path_root = f"{models_dir}/PPO-1"
	print("🧠 Folder structure created...")

	config["logdir"] = logdir
	'''Initializing environment'''
	env = WukongEnv(config)
	# env = DummyVecEnv([lambda: WukongEnv(config)])
	print("🧠 WukongEnv initialized...")
	# Set up visual network
	policy_kwargs = dict(
		features_extractor_class=YOLOFeatureExtractor,  # Replace visual network
		features_extractor_kwargs=dict(features_dim=512)  # Output dimension of visual features
	)


	'''Creating new model or loading existing model'''
	if torch.cuda.is_available():
		device = "cuda"
		print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
	else:
		device = "cpu"
		print("CUDA not available. Using CPU.")

	prev_steps = 57300
	model_path = f"{models_dir}/PPO-1/PPO_huxianfeng_" + str(prev_steps) + ".zip"

	if CREATE_NEW_MODEL:
		model = PPO('MultiInputPolicy',
							env,
							policy_kwargs=policy_kwargs,
							tensorboard_log=logdir,
							n_steps=HORIZON_WINDOW,
							verbose=0,
							device=device)	#Set training device here.
		print("🧠 New Model created...")
	else:
		model = PPO.load(model_path, env=env)
		print("🧠 Model loaded...")

	callback = DetailedLossCallback(model_path_root, prev_steps)
	'''Training loop'''
	while True:
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", log_interval=1,
					callback=callback)
		model.save(f"{models_dir}/PPO-1")
		print(f"🧠 Model updated...")
		obs = env.reset()