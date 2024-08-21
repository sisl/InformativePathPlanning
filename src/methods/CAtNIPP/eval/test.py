import sys
import os

import copy
import csv
import ray
import torch
import time
from multiprocessing import Pool
import numpy as np
import time
from attention_net import AttentionNet
from runner import Runner
from test_worker import WorkerTest
from test_parameters import *

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA backend for computations.")
else:
    device = torch.device("cpu")
    print("Using CPU backend for computations.")


try:
    from runner import Runner
    print("Runner imported successfully!")
except ModuleNotFoundError as e:
    print(f"Failed to import Runner: {e}")


def run_test(test_number):
    time0 = time.time()
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # The device is already determined at the start, so we don't need to reassign it here
    local_device = device  # use the same device for local operations

    global_network = AttentionNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    # print current path
    print("Current path:", os.getcwd())
    checkpoint = torch.load(f'{model_path}/checkpoint.pth', map_location=device, weights_only=True)  # Load checkpoint on the correct device
    global_network.load_state_dict(checkpoint['model'])

    print(f'Loading model: {FOLDER_NAME}...')
    print(f'Total budget range: {BUDGET_RANGE}')

    # Initialize meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]
    weights = global_network.to(local_device).state_dict()
    curr_test = 1
    metric_name = ['budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'MI', 'F1Score', 'cov_trace', 'planning_time']
    perf_metrics = {n: [] for n in metric_name}
    cov_trace_list = []
    time_list = []
    episode_number_list = []
    budget_history = []
    obj_history = []
    obj2_history = []

    try:
        while True:
            jobList = []
            for i, meta_agent in enumerate(meta_agents):
                jobList.append(meta_agent.job.remote(weights, curr_test, budget_range=BUDGET_RANGE, sample_length=SAMPLE_LENGTH, test_number=test_number))
                # jobList.append(meta_agent.job.remote(weights, curr_test, budget_range=BUDGET_RANGE, sample_length=SAMPLE_LENGTH))
                curr_test += 1
            done_id, jobList = ray.wait(jobList, num_returns=NUM_META_AGENT)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, info = job
                episode_number_list.append(info['episode_number'])
                cov_trace_list.append(metrics['cov_trace'])
                time_list.append(metrics['planning_time'])
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])
                budget_history += metrics['budget_history']
                obj_history += metrics['obj_history']
                obj2_history += metrics['obj2_history']

            if curr_test > NUM_TEST:
                print('#Test sample:', NUM_SAMPLE_TEST, '|#Total test:', NUM_TEST, '|Budget range:', BUDGET_RANGE, '|Sample size:', SAMPLE_SIZE, '|K size:', K_SIZE)
                print('Avg time per test:', (time.time() - time0) / NUM_TEST)
                perf_data = [np.nanmean(perf_metrics[n]) for n in metric_name]
                for i, name in enumerate(metric_name):
                    print(name, ':\t', perf_data[i])

                idx = np.array(episode_number_list).argsort()
                cov_trace_list = np.array(cov_trace_list)[idx]
                time_list = np.array(time_list)[idx]

                if SAVE_TRAJECTORY_HISTORY:
                    idx = np.array(budget_history).argsort()
                    budget_history = np.array(budget_history)[idx]
                    obj_history = np.array(obj_history)[idx]
                    obj2_history = np.array(obj2_history)[idx]

                break

        Budget = int(perf_data[0]) + 1
        if SAVE_CSV_RESULT:
            if TRAJECTORY_SAMPLING:
                csv_filename = f'result/CSV/Budget_{Budget}_ts_{PLAN_STEP}_{NUM_SAMPLE_TEST}_{SAMPLE_SIZE}_{K_SIZE}_results.csv'
                csv_filename3 = f'result/CSV3/Budget_{Budget}_ts_{PLAN_STEP}_{NUM_SAMPLE_TEST}_{SAMPLE_SIZE}_{K_SIZE}_planning_time.csv'
            else:
                csv_filename = f'result/CSV/Budget_{Budget}_greedy_{SAMPLE_SIZE}_{K_SIZE}_results.csv'
                csv_filename3 = f'result/CSV3/Budget_{Budget}_greedy_{SAMPLE_SIZE}_{K_SIZE}_planning_time.csv'
            csv_data = [cov_trace_list]
            csv_data3 = [time_list]
            with open(csv_filename, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_data)
            with open(csv_filename3, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_data3)

        if SAVE_TRAJECTORY_HISTORY:
            if TRAJECTORY_SAMPLING:
                csv_filename2 = f'result/CSV2/Budget_{Budget}_ts_{PLAN_STEP}_{NUM_SAMPLE_TEST}_{SAMPLE_SIZE}_{K_SIZE}_trajectory_result.csv'
            else:
                csv_filename2 = f'result/CSV2/Budget_{Budget}_greedy_{SAMPLE_SIZE}_{K_SIZE}_trajectory_result.csv'
            new_file = not os.path.exists(csv_filename2)
            field_names = ['budget', 'obj', 'obj2']
            with open(csv_filename2, 'a') as csvfile:
                writer = csv.writer(csvfile)
                if new_file:
                    writer.writerow(field_names)
                csv_data = np.concatenate((budget_history.reshape(-1, 1), obj_history.reshape(-1, 1), obj2_history.reshape(-1, 1)), axis=-1)
                writer.writerows(csv_data)

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


@ray.remote(num_cpus=8/NUM_META_AGENT, num_gpus=0)  # Set num_gpus to 0 as MPS doesn't use CUDA
class RLRunner(Runner):
    def __init__(self, metaAgentID):
        super().__init__(metaAgentID)
        self.device = device  # Use the device defined earlier

    def singleThreadedJob(self, episodeNumber, budget_range, sample_length, test_number):
        save_img = episodeNumber % SAVE_IMG_GAP == 0
        # uncomment the following line to make the test deterministic
        # np.random.seed(SEED + 100 * episodeNumber)
        # worker = WorkerTest(self.metaAgentID, self.localNetwork, episodeNumber, budget_range, sample_length, self.device, save_image=save_img, greedy=False, seed=SEED + 100 * episodeNumber, test_number=test_number)
        np.random.seed(SEED + 100 * episodeNumber + test_number)
        worker = WorkerTest(self.metaAgentID, self.localNetwork, episodeNumber, budget_range, sample_length, self.device, save_image=save_img, greedy=False, seed=SEED + 100 * episodeNumber + test_number, test_number=test_number)
        # worker = WorkerTest(self.metaAgentID, self.localNetwork, episodeNumber, budget_range, sample_length, self.device, save_image=save_img, greedy=False, seed=SEED + 100 * episodeNumber)
        worker.work(episodeNumber, 0)
        perf_metrics = worker.perf_metrics
        
        ground_truth = worker.env.ground_truth

        return perf_metrics

    def multiThreadedJob(self, episodeNumber, budget_range, sample_length):
        save_img = True if (SAVE_IMG_GAP != 0 and episodeNumber % SAVE_IMG_GAP == 0) else False
        #save_img = False
        np.random.seed(SEED + 100 * episodeNumber)
        #torch.manual_seed(SEED + 100 * episodeNumber)
        worker = WorkerTest(self.metaAgentID, self.localNetwork, episodeNumber, budget_range, sample_length, self.device, save_image=save_img, greedy=False, seed=SEED + 100 * episodeNumber)
        subworkers = [copy.deepcopy(worker) for _ in range(NUM_SAMPLE_TEST)]
        p = Pool(processes=NUM_SAMPLE_TEST)
        results = []
        for testID, subw in enumerate(subworkers):
            results.append(p.apply_async(subw.work, args=(episodeNumber, testID+1)))
        p.close()
        p.join()
        all_results = []
        best_score = np.inf
        perf_metrics = None
        for res in results:
            metric = res.get()
            all_results.append(metric)
            if metric['cov_trace'] < best_score: # TODO
                perf_metrics = metric
                best_score = metric['cov_trace']
        return perf_metrics

    def job(self, global_weights, episodeNumber, budget_range, sample_length=None, test_number=None):
        self.set_weights(global_weights)
        metrics = self.singleThreadedJob(episodeNumber, budget_range, sample_length, test_number)

        info = {
            "id": self.metaAgentID,
            "episode_number": episodeNumber,
            "test_number": test_number
        }

        return metrics, info


if __name__ == '__main__':
    ray.init()
    for i in range(100):
        print("#############################################")
        print(f"Running test {i}")
        print("#############################################")
        run_test(i)