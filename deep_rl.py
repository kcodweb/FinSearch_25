# Install compatible Gym, TensorFlow, and plotting library
!pip install gym==0.25.2 tensorflow matplotlib

# Patch NumPy for Gym compatibility
import numpy as np
np.bool8 = np.bool_

# Imports
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Environment setup
env = gym.make("Pendulum-v1")
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

# Ornstein-Uhlenbeck noise for exploration
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt)
              * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x
    def reset(self):
        self.x_prev = (
            self.x_initial
            if self.x_initial is not None
            else np.zeros_like(self.mean)
        )

# Replay buffer for experience replay
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((buffer_capacity, num_states))
        self.action_buffer = np.zeros((buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((buffer_capacity, 1))
        self.next_state_buffer = np.zeros((buffer_capacity, num_states))
    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1
    def sample(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        indices = np.random.choice(record_range, self.batch_size)
        state_batch = tf.convert_to_tensor(self.state_buffer[indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[indices])
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[indices])
        return state_batch, action_batch, reward_batch, next_state_batch

# Actor model
def get_actor():
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="tanh")(out)
    outputs = outputs * upper_bound
    return keras.Model(inputs, outputs)

# Critic model
def get_critic():
    state_in = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_in)
    action_in = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(16, activation="relu")(action_in)
    concat = layers.Concatenate()([state_out, action_out])
    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)
    return keras.Model([state_in, action_in], outputs)

# Soft update of target network weights
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

# Hyperparameters
std_dev = 0.2
actor_lr = 0.001
critic_lr = 0.002
total_episodes = 100
gamma = 0.99
tau = 0.005
buffer = Buffer(50000, 64)
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=std_dev * np.ones(1))

# Create networks
actor_model = get_actor()
critic_model = get_critic()
target_actor = get_actor()
target_critic = get_critic()
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

actor_optimizer = keras.optimizers.Adam(actor_lr)
critic_optimizer = keras.optimizers.Adam(critic_lr)

# Policy to pick action, with exploration noise
def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    sampled = sampled_actions.numpy() + noise
    legal = np.clip(sampled, lower_bound, upper_bound)
    return [np.squeeze(legal)]

# Training step (actor & critic update)
@tf.function
def update(state_batch, action_batch, reward_batch, next_state_batch):
    reward_batch = tf.cast(reward_batch, tf.float32)
    # Critic update
    with tf.GradientTape() as tape:
        target_actions = target_actor(next_state_batch, training=True)
        y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True)
        critic_value = critic_model([state_batch, action_batch], training=True)
        critic_loss = tf.reduce_mean(tf.square(y - critic_value))
    critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))
    # Actor update
    with tf.GradientTape() as tape:
        actions = actor_model(state_batch, training=True)
        critic_value = critic_model([state_batch, actions], training=True)
        actor_loss = -tf.reduce_mean(critic_value)
    actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

# Training loop
ep_rewards = []
avg_rewards = []

for ep in range(total_episodes):
    prev_state = env.reset()
    episodic_reward = 0
    while True:
        tf_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = policy(tf_state, ou_noise)
        next_state, reward, done, _ = env.step(action)
        buffer.record((prev_state, action, reward, next_state))
        episodic_reward += reward
        if buffer.buffer_counter > buffer.batch_size:
            s, a, r, ns = buffer.sample()
            update(s, a, r, ns)
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)
        if done:
            break
        prev_state = next_state
    ep_rewards.append(episodic_reward)
    avg = np.mean(ep_rewards[-10:])
    avg_rewards.append(avg)
    print(f"Episode {ep+1}: Reward = {episodic_reward:.2f}, Avg(10) = {avg:.2f}")

# Plot learning curve
plt.plot(avg_rewards)
plt.xlabel("Episode")
plt.ylabel("Average Reward (last 10)")
plt.title("DDPG on Pendulum-v1")
plt.grid(True)
plt.show()
