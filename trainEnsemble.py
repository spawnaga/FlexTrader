from agentsEnsemble import MultiTask
from trader import Trader, Market
from collections import deque


def train():
    levels = {0: 0.005, 1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05, 6: 0.1, 7: 0.15, 8: 0.2, 9: 0.25}
    trader = Trader()
    action_size = 5
    previous_row = None
    steps = [0]
    window_size = 100
    rolling_window = deque(maxlen=window_size)
    current_iteration = 0
    batch_size = 10
    market = Market(trader)
    current_batch_size_level = 0
    market.update_data()
    market.get_df()
    df = market.get_df()
    state = market.get_state(0)
    agent = MultiTask(action_size=action_size, state_size=state.shape[1])
    replay_functions = {
        "dqn": agent.replay_dqn,
        "ddqn": agent.replay_ddqn,
        "actor_critic": agent.replay_actor_critic,
        "policy_gradient": agent.replay_policy_gradient
    }
    agent.load_models_and_memories('model2')
    agent.load_performance_and_counters()
    for _, task in agent.tasks.items():
        if eval(f'agent.{task}_memory._size()') == 0:
            agent.load_models_and_memories('model2')
        for level, percentage in levels.items():
            if int(len(df) * percentage) < eval(f'agent.{task}_memory._size()'):
                batch_size = int(len(df) * percentage)
    while not trader.profit >= 1000000 * 0.3 or not trader.num_trades >= 1000:
        current_iteration += 1
        market = Market(trader)
        market.update_data()
        df = market.get_df()
        state = market.get_state(i=0)

        episode_reward = 0

        for i, row in df.iterrows():
            if previous_row is None:
                previous_row = row
            done = i + 2 >= len(df)
            if done:
                break
            if i == int(len(df) * levels[current_batch_size_level]) and batch_size <= len(df) * levels[
                current_batch_size_level]:
                batch_size = int(len(df) * levels[current_batch_size_level])
                print(
                    f'Level {list(levels.keys())[current_batch_size_level]} is done. Batch size now is {batch_size} '
                    f'({levels[current_batch_size_level] * 100}% of the data)')
                if current_batch_size_level <= next(reversed(levels.items()))[0] - 1:
                    current_batch_size_level += 1

            task = agent.choose_task(state, 'train')
            action = agent.act(task=task, state=state, job='train')
            reward = trader.trade(action, row, previous_row, i)
            next_state = market.get_state(i + 1, numContracts=trader.num_contracts)
            steps.append(i)
            agent.add_master_transition(state, task, reward, next_state, done)
            for _, task1 in agent.tasks.items():
                agent.add_to_memory(task1, state, action, reward, next_state, done)
            previous_row = row
            state = next_state
            rolling_window.append(trader.realized_profit_loss)
            episode_reward += reward
            agent.replay_master(batch_size)
            replay_functions[agent.tasks[task]](batch_size)

            if i % 50 == 0:
                agent.save_models_and_memories('model2')
                print(f"Episode {i}/{current_iteration}: final account value {trader.total_value}")
                print(f" which is a total profit/loss of {trader.total_value - trader.capital}")


if __name__ == '__main__':
    train()
