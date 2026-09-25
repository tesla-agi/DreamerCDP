import numpy as np

class ReplayBuffer:
    def __init__(self,obs_shape=(64,64,3),max_episodes=100,max_steps=500):
        self.obs_shape=obs_shape
        self.max_episodes=max_episodes
        self.max_steps=max_steps

        self.observations=np.zeros(
            (self.max_episodes,self.max_steps+1,*self.obs_shape),dtype=np.uint8
        )

        self.actions=np.zeros(
            (self.max_episodes,self.max_steps),dtype=np.int64
        )

        self.rewards=np.zeros(
            (self.max_episodes,self.max_steps),dtype=np.float32
        )

        self.episode_lengths=np.zeros(
            self.max_episodes,dtype=np.int64
        )

        self.continue_pred=np.zeros(
            (self.max_episodes,self.max_steps),dtype=np.float32
        )

        self.num_episodes=0
        self.write_idx=0

    def add_episode(self,obs_list,action_list,reward_list,done_list=None):
        idx=self.write_idx
        T=len(action_list)
        if done_list is None:
            self.continue_pred[idx,:T]=1.0

        else:
            self.continue_pred[idx,:T]=1.0-np.array(done_list,dtype=np.float32)

        self.observations[idx,:T+1]=np.array(obs_list,dtype=np.uint8)
        self.actions[idx,:T]=np.array(action_list,dtype=np.int64)
        self.rewards[idx,:T]=np.array(reward_list,dtype=np.float32)
        self.episode_lengths[idx]=T
        self.write_idx=(self.write_idx+1)%self.max_episodes
        self.num_episodes=min(self.num_episodes+1,self.max_episodes)

    def sample_sequence(self,batch_size=50,seq_len=50):
        obs_batch=np.zeros(
            (batch_size,seq_len+1,*self.obs_shape),dtype=np.uint8
        )
        action_batch=np.zeros(
            (batch_size,seq_len),dtype=np.int64
        )
        reward_batch=np.zeros(
            (batch_size,seq_len),dtype=np.float32
        )
        continue_batch=np.zeros(
            (batch_size,seq_len),dtype=np.float32
        )

        for idx in range(batch_size):
            while True:
                ep_idx=np.random.randint(0,self.num_episodes)
                if self.episode_lengths[ep_idx]>=seq_len:
                    break

            max_start=self.episode_lengths[ep_idx]-seq_len
            start_indices=np.random.randint(0,max_start+1)
            end=start_indices+seq_len

            obs_batch[idx]=self.observations[ep_idx,start_indices:end+1]
            action_batch[idx]=self.actions[ep_idx,start_indices:end]
            reward_batch[idx]=self.rewards[ep_idx,start_indices:end]
            continue_batch[idx]=self.continue_pred[ep_idx,start_indices:end]

        return obs_batch,action_batch,reward_batch,continue_batch


    def save(self,path):
        np.savez(path,
            observations=self.observations[:self.num_episodes],
            actions=self.actions[:self.num_episodes],
            rewards=self.rewards[:self.num_episodes],
            num_episodes=self.num_episodes,
            episode_lengths=self.episode_lengths[:self.num_episodes],
            continue_=self.continue_pred[:self.num_episodes],
        )
        print(f"Saved {self.num_episodes} episodes to {path}")

    def load(self,path):
        data=np.load(path)
        self.num_episodes=int(data["num_episodes"])
        self.episode_lengths[:self.num_episodes]=data["episode_lengths"]
        self.observations[:self.num_episodes]=data["observations"]
        self.actions[:self.num_episodes]=data["actions"]
        self.rewards[:self.num_episodes]=data["rewards"]
        if "continue_" in data:
            self.continue_pred[:self.num_episodes]=data["continue_"]
        else:
            for i in range(self.num_episodes):
                self.continue_pred[i,:self.episode_lengths[i]]=1.0
        print(f"Loaded {self.num_episodes} episodes from {path}")

    def __len__(self):
        return self.num_episodes


if __name__ == "__main__":
    buf = ReplayBuffer(obs_shape=(2, 2, 3), max_episodes=3, max_steps=10)
    for ep in range(5):
        T = 10
        obs = [np.full((2, 2, 3), ep, dtype=np.uint8)] * (T + 1)
        buf.add_episode(obs, [0] * T, [float(ep)] * T, [False] * T)
    print("num_episodes:", buf.num_episodes, " (expect 3)")
    print("write_idx:   ", buf.write_idx, " (expect 2)")
    print("slot rewards:", buf.rewards[:, 0], " (expect [3. 4. 2.])")