import os
import time
from cec2013.cec2013 import *
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

EPS = 1e-5

# Problem parameters
times_of_run = 51  # Number of runs for each function
fun_to_run = list(range(1, 21))  # Function set
fun_num = 20  # Total function number
# population_size_set1 = [5, 80, 5, 5, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
population_size_set1 = [80, 80, 80, 80, 80, 100, 300, 300, 300, 100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
population_size_set = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
population_size_set2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# population_size_set = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
# population_size_set = [5, 5, 5, 5, 5, 30, 50, 50, 50, 5, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
# population_size_set = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
# population_size_set = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
sp_size_set= [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 10, 10]

epsilon_set_size = 5
epsilon_set = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]  # 判断精度
record_size = 10

# params for method
# fw_size = 5
sp_size = 100
gm_ratio = 0.5
parameter_N = 2
parameter_b = 1.5

# Initialize data structures
runs_global_num = np.zeros([epsilon_set_size, times_of_run])
runs_used_fitness = np.zeros([epsilon_set_size, times_of_run])
# epsilon_flag = np.zeros([epsilon_set_size, 1])
# record_fes = np.zeros([record_size, 1])
runs_global_num_vs_fes = np.zeros([epsilon_set_size, times_of_run, record_size])


# Function to evaluate the population
def evaluate_population(population, func, fes, population_size):
    # results = [func.evaluate(ind) for ind in population]
    results = np.zeros([population_size])
    for i in range(population_size):
        results[i] = func.evaluate(population[i])
    fes += population_size
    return results, fes


def evaluate_idv(idv, func, fes):
    result = func.evaluate(idv)
    if isinstance(result, (np.ndarray, list)) and len(result) == 1:
        result = result.item()
    fes += 1
    return result, fes


def _map(samples, firework, amp, lower_bound, upper_bound):
    # 判断 samples 是否在范围内
    in_bound = (samples > lower_bound) * (samples < upper_bound)
    # 初始化随机样本矩阵
    rand_samples = np.zeros_like(samples)
    # 针对每个维度生成随机样本
    for i in range(len(firework)):
        lower_bound_value = max(firework[i] - amp, lower_bound[i])
        upper_bound_value = min(firework[i] + amp, upper_bound[i])
        # 检查是否超出范围
        if upper_bound_value - lower_bound_value > np.finfo(np.float64).max:
            raise ValueError(
                f"Range for random uniform exceeds valid bounds for index {i}: [{lower_bound_value}, {upper_bound_value}]")
        rand_samples[:, i] = np.random.uniform(max(firework[i] - amp, lower_bound[i]),
                                               min(firework[i] + amp, upper_bound[i]),
                                               len(samples))
    # 更新 samples，确保样本在范围内
    samples = in_bound * samples + (1 - in_bound) * rand_samples

    return samples


def _map2(samples, dimension, lower_bound, upper_bound):

    # 针对每个维度生成随机样本
    for i in range(dimension):
        if samples[i] < lower_bound[i]:
            samples[i] = lower_bound[i]
        if samples[i] > upper_bound[i]:
            samples[i] = upper_bound[i]

    return samples


# Function to compute global optima found
def compute_global_optima_found(population, results, epsilon, func):
    # 检查 population 是否为空
    if not population.ndim:
        # raise ValueError("Population is 1-ndim.")
        print('cao')
    # elif population.ndim == 0:
    #     # raise ValueError("Population is empty.")
    #     count = 0
    #     seeds = []
    #     return count, seeds
    # else:
    count, seeds, seeds_fits = how_many_goptima(population, results, func, epsilon)
    return count, seeds, seeds_fits


# Function to evolve the population (simplified for this example)
def evolve_population(population, results, population_size, dimension, amps, func, fes, lbound, ubound, init_amp,
                      sp_size_, max_fes, sp_size_pop, r_rate):
    all_sparks = []
    all_spark_fits = []

    for idx in range(population_size):
        # dynamic amps
        if amps[idx] == 0:
            amps[idx] = init_amp * np.random.uniform(0.5, 1)
            rand_sample = np.random.uniform(lbound, ubound, (1, dimension))
            population[idx] = rand_sample[0]
            results[idx], fes = evaluate_idv(population[idx, :], func, fes)

    # compute explode sparks
    num_sparks = [int(sp_size_)] * population_size

    # explode - 使用虚拟种群DE变异（不计算虚拟个体fitness）
    e_sparks = []
    e_fits = []
    for idx in range(population_size):
        sparks = []

        # 根据当前半径自适应调整搜索策略
        current_amp_ratio = amps[idx] / init_amp
        # for j in range(num_sparks[idx]):
        # 大半径时：偏向探索（类似原随机偏移）
        # 小半径时：偏向开发（标准DE）
        # if current_amp_ratio > 0.5:  # 大半径阶段
        #     # 混合策略：50% DE + 50% 随机
        #     if np.random.rand() < 0.5:
        #         # DE变异 - 只生成虚拟个体进行DE操作，不计算其fitness
        #         VX_i1 = generate_virtual_individual(population[idx, :], amps[idx], dimension, lbound, ubound)
        #         VX_i2 = generate_virtual_individual(population[idx, :], amps[idx], dimension, lbound, ubound)
        #         F = 0.8  # 大半径用大F，增强探索
        #         mutant = population[idx, :] + F * (VX_i1 - VX_i2)
        #     else:
        #         # 保持原随机偏移
        #         bias = np.random.uniform(-1, 1, dimension)
        #         mutant = population[idx, :] + bias * amps[idx]
        # else:  # 小半径阶段
        # 纯DE变异 - 只生成虚拟个体进行DE操作，不计算其fitness
        # 突变率CR
        CR = 0.8

        # base_vector = generate_virtual_individual(population[idx, :], amps[idx], dimension, lbound, ubound)
        for j in range(num_sparks[idx]):
            base_vector = generate_virtual_individual(population[idx, :], amps[idx], dimension, lbound, ubound)

            # 在粒球内随机选择一个点作为DE变异的基向量
            # 生成两个不同的虚拟个体
            VX_i1 = generate_virtual_individual(population[idx, :], amps[idx], dimension, lbound, ubound)
            VX_i2 = generate_virtual_individual(population[idx, :], amps[idx], dimension, lbound, ubound)

            # 确保两个虚拟个体不同
            while np.array_equal(VX_i1, VX_i2):
                VX_i2 = generate_virtual_individual(population[idx, :], amps[idx], dimension, lbound, ubound)

            F = 0.1  # 缩放因子

            # DE变异：在随机基向量上进行
            mutant_vector = base_vector + F * (VX_i1 - VX_i2)

            # DE交叉操作
            trial_vector = np.copy(population[idx, :])  # 从当前个体开始
            cross_points = np.random.rand(dimension) < CR  # 随机选择交叉维度

            # 确保至少一个维度交叉（CR=0.9时通常会有多个维度交叉）
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimension)] = True

            trial_vector[cross_points] = mutant_vector[cross_points]

            mutant = _map2(trial_vector, dimension, lbound, ubound)
            sparks.append(mutant)

        sparks = np.array(sparks)
        # 只计算最终变异后个体的fitness，不计算虚拟个体的fitness
        spark_fits, fes = evaluate_population(sparks, func, fes, num_sparks[idx])
        e_sparks.append(sparks)
        e_fits.append(spark_fits)
    # mutate (保持原有的引导火花机制)
    m_sparks = []
    m_fits = []
    for idx in range(population_size):
        sparks = []
        top_num = int(num_sparks[idx] * gm_ratio)  # make a sgd vector
        sort_idx = np.argsort(e_fits[idx])
        top_idx = sort_idx[-top_num:]
        btm_idx = sort_idx[:top_num]

        top_mean = np.mean(e_sparks[idx][top_idx, :], axis=0)
        btm_mean = np.mean(e_sparks[idx][btm_idx, :], axis=0)
        delta = top_mean - btm_mean  # Max_problem

        weight = np.random.uniform(0, parameter_b, (parameter_N, 1))
        sparks = population[idx, :] + delta * weight  # Max_problem,center to add
        sparks = _map(sparks, population[idx, :], amps[idx], lbound, ubound)
        m_fit, fes = evaluate_population(sparks, func, fes, parameter_N)
        m_sparks.append(sparks)
        m_fits.append(m_fit)

    # select
    n_fireworks = np.empty([population_size, dimension])
    n_fits = np.empty([population_size])
    for idx in range(population_size):
        sparks = np.concatenate([population[idx, :][np.newaxis, :],
                                 e_sparks[idx],
                                 m_sparks[idx]], axis=0)
        spark_fits = np.concatenate([np.array([results[idx]]),
                                     e_fits[idx],
                                     m_fits[idx]], axis=0)
        min_idx = np.argmax(spark_fits)  # Max_problem

        n_fireworks[idx, :] = sparks[min_idx, :]
        n_fits[idx] = spark_fits[min_idx]

        if n_fits[idx] > results[idx]:
            # 成功改进，保持或稍微增大搜索范围
            pass
        else:
            # 未改进，缩小搜索范围
            amps[idx] *= 0.9

        all_sparks.append(sparks)
        all_spark_fits.append(spark_fits)

    return n_fireworks, n_fits, all_sparks, all_spark_fits, fes

def generate_virtual_individual(individual, radius, dimension, lbound, ubound):
    """
    为给定个体生成虚拟个体
    根据DIDE论文中的公式：VX_i^d = Random(max(X_i^d - R_i^d/2, L^d), min(X_i^d + R_i^d/2, U^d))
    """
    virtual_individual = np.zeros(dimension)

    for d in range(dimension):
        lower_bound = max(individual[d] - radius / 2, lbound[d])
        upper_bound = min(individual[d] + radius / 2, ubound[d])
        virtual_individual[d] = np.random.uniform(lower_bound, upper_bound)

    return virtual_individual


def overlap_elimination(temp_population, temp_results,
                      best_fit, results, new_population, new_results, amps, population_size, max_iter, dimension,
                      num_iter, lbound, ubound, init_amp, func, fes, r_rate, loop_times):

    # restart
    all_sparks = []


def restart_operation(temp_population, temp_results,
                      best_fit, results, new_population, new_results, amps, population_size, max_iter, dimension,
                      num_iter, lbound, ubound, init_amp, func, fes, r_rate, loop_times, rho):
    # # restart
    # improves = new_results - results
    # improves1 = best_fit - new_results
    # min_fit = best_fit          # Max_problem
    # restart = (improves > EPS) * (improves * (max_iter - num_iter) < (min_fit - new_results))

    # # restart：
    # improves = best_fit - new_results
    # min_fit = best_fit          # Max_problem
    # # restart = (improves > 0) * (improves * (max_iter - num_iter) < (min_fit - new_results))
    # restart1 = (improves > 1e-1)
    #
    # restart = (pop_re_count == 3)   # Any else pra?

    # # restart
    # # 分别存储最大值和最小值的数组
    # max_values = []
    # min_values = []
    # # 遍历每个子列表，找到最大值和最小值，并添加到相应的数组中
    # for lst in temp_results:
    #     max_values.append(max(lst))
    #     min_values.append(min(lst))
    # max_values = np.array(max_values)
    # min_values = np.array(min_values)
    # improves = new_results - results
    # improves_global = best_fit - new_results
    # min_fit = best_fit          # Max_problem
    # # restart_satisfied = (improves > 0) * (improves < EPS)
    # # restart = restart_satisfied
    # restart_satisfied = (improves > 0) * (improves < EPS) * (improves_global > 0) * (improves_global < EPS)
    # # restart_local_optima = (improves > 0) * (improves < EPS) * (improves_global > 0) * (improves_global > 1e-1)
    # restart = restart_satisfied

    # restart：radius < 1e-5
    cha = new_results[:, np.newaxis] - np.array(temp_results)
    # 计算每一行的均值
    # row_means = np.mean(cha, axis=1)
    row_means = np.max(cha, axis=1)
    chase = best_fit - new_results
    cur_var = np.var(cha, axis=1)
    # restart = (row_means < 1e-5)
    # restart = (row_means < 1e-5) * (chase > 1e-1) + (row_means < 1e-5) * (chase < 1e-5)
    # restart = (row_means < 1e-5) + (row_means < 1e-1) * (chase > 1)
    restart = (row_means < 1e-5) + (cur_var < 1e-5) * (chase > 1e-1)

    # restart：radius < 1e-5
    # improves = amps
    # min_fit = best_fit  # Max_problem
    # restart = (improves <= 1e-8)

    # # 半径又小，值又不好
    # # Determine which individuals to restart based on radius and fitness thresholds
    # radius_condition = improves < (1/3 * init_amp)
    # fitness_condition = (best_fit - new_results) > 1e-1
    # restart = radius_condition & fitness_condition

    # # restart：probability choose
    # r_bilv = amps / init_amp
    # select_r_bilv = r_bilv ** 2
    # # 计算基于适应值的选择概率
    # min_fit = np.min(new_results)
    # max_fit = np.max(new_results)
    # selection_prob_fit = (new_results - min_fit + 1e-4) / (max_fit - min_fit + 1e-4)

    # # 找到适应值最好的个体的索引
    # best_index = np.argmax(new_results)     # Max_problem
    # best_individual = new_population[best_index]
    # # 计算每个个体与适应值最好的个体之间的欧式距离
    # distances = np.linalg.norm(new_population - best_individual, axis=1)
    # # 计算距离最近和最远的个体之间的距离
    # min_dist = np.min(distances[distances != 0])  # 排除距离为 0 的情况，即最好的个体自己
    # max_dist = np.max(distances)
    # # 计算基于欧式距离的选择概率
    # selection_prob_dist = (distances - min_dist + 1e-4) / (max_dist - min_dist + 1e-4)

    # # 初始化重启标志
    # restart = np.zeros(population_size, dtype=bool)
    # # 对每个个体计算是否重启
    # for i in range(population_size):
    #     if np.random.rand() > selection_prob_fit[i]:
    #         if np.random.rand() > select_r_bilv[i]:
    #             restart[i] = True

    # 首先创建距离矩阵（避免重复计算）
    dist_matrix = cdist(new_population, new_population)  # shape: (N, N)
    # 只考虑上三角（去除对角线），避免重复检查
    N = len(new_population)
    for i in range(N):
        if restart[i]:
            continue
        for j in range(i + 1, N):
            if dist_matrix[i][j] <= 1:
                restart[j] = True

    replace = restart[:, np.newaxis].astype(np.int32)
    restart_num = int(sum(replace))

    if restart_num > 0:
        rand_sample = np.random.uniform(lbound,
                                        ubound,
                                        (population_size, dimension))
        new_population = (1 - replace) * new_population + replace * rand_sample
        new_results[restart], fes = evaluate_population(new_population[restart, :], func, fes, restart_num)
        # amps[restart] = init_amp
        amps[restart] = init_amp * np.random.uniform(0.5, 1, population_size)[restart]
        r_rate[restart] = 0.9

    return new_population, new_results, fes


def gaussian_lss(best_fit, best_idv, global_optima_fit_list, global_optima_list, population_size, dimension,
                lbound, ubound, func, fes, max_fes, amps, init_amp,
                new_population, new_results):

    # restart：probability choose
    # 计算基于适应值的选择概率
    # improve = best_fit - global_optima_fit_list
    improve = global_optima_fit_list
    min_fit = min(improve)
    max_fit = max(improve)
    selection_prob_fit = (max_fit - improve + 1e-4) / (max_fit - min_fit + 1e-4)

    all_v = np.array(global_optima_list)
    # 计算每个个体与适应值最好的个体之间的欧式距离
    distances = np.linalg.norm(all_v - best_idv, axis=1)
    # 计算距离最近和最远的个体之间的距离
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    # 计算基于欧式距离的选择概率
    selection_prob_dist = (distances - min_dist + 1e-4) / (max_dist - min_dist + 1e-4)

    # 对每个个体执行局部搜索
    for i in range(len(global_optima_fit_list)):
        sigma = 1
        if improve[i] < 1e-5:
            continue
        elif 1e-5 < improve[i] < 1e-4:
            sigma = 1e-4
        elif 1e-4 < improve[i] < 1e-3:
            sigma = 1e-3
        elif 1e-3 < improve[i] < 1e-2:
            sigma = 1e-2
        elif 1e-2 < improve[i] < 1e-1:
            sigma = 1e-1
        # if np.random.rand() < selection_prob_fit[i]:
            # if np.random.rand() < selection_prob_dist[i]:
            # if np.random.rand() < selection_prob_fit[i]:
        ls_flag = True
        # sigma = 1e-1
        ls_points_nums = 2
        while ls_flag:
            # bias = np.random.uniform(-1, 1, [ls_points_nums, dimension])  # 随机取点
            bias = np.random.normal(0, 1, [ls_points_nums, dimension])  # 高斯取点
            new_point = global_optima_list[i] + bias * sigma
            new_point = _map(new_point, global_optima_list[i], sigma, lbound, ubound)
            new_point_fitness, fes = evaluate_population(new_point, func, fes, ls_points_nums)
            if np.min(new_point_fitness) > global_optima_fit_list[i]:
                idx = np.argmax(new_point_fitness)
                global_optima_list[i] = new_point[idx]
                global_optima_fit_list[i] = new_point_fitness[idx]
            # else:
            sigma *= 0.1
            if sigma < 1e-5:
                ls_flag = False

    return global_optima_fit_list, global_optima_list, fes, new_population, new_results


def gaussian_ls(best_fit, best_idv, global_optima_fit_list, global_optima_list, population_size, dimension,
                lbound, ubound, func, fes, max_fes, amps, init_amp,
                new_population, new_results):

    # restart：probability choose
    # 计算基于适应值的选择概率
    # improve = best_fit - global_optima_fit_list
    improve = global_optima_fit_list
    min_fit = min(improve)
    max_fit = max(improve)
    selection_prob_fit = (max_fit - improve + 1e-4) / (max_fit - min_fit + 1e-4)

    all_v = np.array(global_optima_list)
    # 计算每个个体与适应值最好的个体之间的欧式距离
    distances = np.linalg.norm(all_v - best_idv, axis=1)
    # 计算距离最近和最远的个体之间的距离
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    # 计算基于欧式距离的选择概率
    selection_prob_dist = (distances - min_dist + 1e-4) / (max_dist - min_dist + 1e-4)

    # 对每个个体执行局部搜索
    for i in range(len(global_optima_fit_list)):
        sigma = 1
        if improve[i] < 1e-5:
            continue
        else:
            sigma = 1e-1
        # if np.random.rand() < selection_prob_fit[i]:
            # if np.random.rand() < selection_prob_dist[i]:
            # if np.random.rand() < selection_prob_fit[i]:
        ls_flag = True
        # sigma = 1e-1
        ls_points_nums = 2
        while ls_flag:
            # bias = np.random.uniform(-1, 1, [ls_points_nums, dimension])  # 随机取点
            bias = np.random.normal(0, 1, [ls_points_nums, dimension])  # 高斯取点
            new_point = global_optima_list[i] + bias * sigma
            new_point = _map(new_point, global_optima_list[i], sigma, lbound, ubound)
            new_point_fitness, fes = evaluate_population(new_point, func, fes, ls_points_nums)
            if np.min(new_point_fitness) > global_optima_fit_list[i]:
                idx = np.argmax(new_point_fitness)
                global_optima_list[i] = new_point[idx]
                global_optima_fit_list[i] = new_point_fitness[idx]
            # else:
            sigma *= 0.1
            if sigma < 1e-5:
                ls_flag = False

    return global_optima_fit_list, global_optima_list, fes, new_population, new_results


def gaussian_ls1(best_fit, best_idv, population_size, dimension,
                lbound, ubound, func, fes, max_fes, amps, init_amp,
                new_population, new_results, temp_population, temp_results):

    # restart：probability choose
    # 计算基于适应值的选择概率
    improve = new_results
    min_fit = min(improve)
    max_fit = max(improve)
    selection_prob_fit = (improve - min_fit + 1e-4) / (max_fit - min_fit + 1e-4)

    all_v = new_population
    # 计算每个个体与适应值最好的个体之间的欧式距离
    distances = np.linalg.norm(all_v - best_idv, axis=1)
    # 计算距离最近和最远的个体之间的距离
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    # 计算基于欧式距离的选择概率
    selection_prob_dist = (distances - min_dist + 1e-4) / (max_dist - min_dist + 1e-4)

    ls_points_nums = 2
    # 对每个个体执行局部搜索
    for i in range(population_size):
        sigma = 1e-1
        ls_flag = True
        while ls_flag:
            if selection_prob_fit[i] == 1:
                bias = np.random.normal(0, 1, [ls_points_nums, dimension])          # 高斯取点
                new_point = new_population[i] + bias * sigma
                new_point = _map(new_point, new_population[i], sigma, lbound, ubound)
                new_point_fitness, fes = evaluate_population(new_point, func, fes, ls_points_nums)
                if np.min(new_point_fitness) > new_results[i]:
                    idx = np.argmax(new_point_fitness)
                    new_population[i] = new_point[idx]
                    new_results[i] = new_point_fitness[idx]
            elif np.random.rand() < selection_prob_fit[i]:
                if np.random.rand() < selection_prob_dist[i]:
                # if np.random.rand() < selection_prob_fit[i]:
                    # bias = np.random.uniform(-1, 1, [ls_points_nums, dimension])  # 随机取点
                    bias = np.random.normal(0, 1, [ls_points_nums, dimension])  # 高斯取点
                    new_point = new_population[i] + bias * sigma
                    new_point = _map(new_point, new_population[i], sigma, lbound, ubound)
                    new_point_fitness, fes = evaluate_population(new_point, func, fes, ls_points_nums)
                    if np.min(new_point_fitness) > new_results[i]:
                        idx = np.argmax(new_point_fitness)
                        new_population[i] = new_point[idx]
                        new_results[i] = new_point_fitness[idx]
            sigma *= 1e-1
            if sigma < 1e-5:
                ls_flag = False

    return fes, new_population, new_results, temp_population, temp_results


# Main program
def main():
    # init random seed
    np.random.seed(int(os.getpid() * time.perf_counter() / 10000))

    for fun in range(fun_num):
    # for fun in [6, 7, 8, 10, 11, 16, 17, 18, 19]:
    # for fun in [19]:
    # 跑多个函数
        print("\n")
        print(f"Function {fun_to_run[fun]} Began!")

        func = CEC2013(fun_to_run[fun])  # Max_problem

        # params for prob
        dimension = func.get_dimension()
        global_optima_num = func.get_no_goptima()
        max_fes = func.get_maxfes() # FEs
        lbound = np.array([func.get_lbound(i) for i in range(dimension)])
        ubound = np.array([func.get_ubound(i) for i in range(dimension)])
        rho = func.get_rho()    # use or not  半径阈值 判断是不是在同一个峰

        init_amp = (ubound - lbound)[0]     # it‘s a question！一维

        # Record function evaluation fes_thresholds
        record_fes = np.linspace(max_fes / record_size, max_fes, record_size).astype(int)

        population_size = population_size_set[fun_to_run[fun] - 1]
        # population_size = 10

        # sp_size_ = int(np.floor(2 + np.log(dimension)))        # fixed size # 子群规模
        # sp_size_ = int(np.floor(1 + np.sqrt(dimension)))  # fixed size
        # sp_size_ = int(dimension + 1)
        # sp_size_ = 10
        sp_size_=sp_size_set[fun_to_run[fun] - 1]
        sp_size_pop = -1    # calculate with dimension、radius

        r_rate = np.ones(population_size) * 0.9  # 重启概率
        loop_times = np.zeros(population_size)

        # Main loop
        # timesOfRun
        for run in range(times_of_run):
            print(f"Running the {run}th time")

            fes = 0
            record_count = 0
            epsilon_flag = [False] * epsilon_set_size

            # Initialize population and fitness
            population = np.random.uniform(lbound, ubound, [population_size, dimension])
            results, fes = evaluate_population(population, func, fes, population_size)
            amps = np.ones(population_size) * init_amp  # 每个个体的半径幅度
            amps = amps * np.random.uniform(0.5, 1, population_size)      # different init granular-ball radius
            best_idx = np.argmax(results)  # Max problem
            best_idv = population[best_idx, :]
            best_fit = results[best_idx]

            # Start main loop

            # 创建一个解集列表，存储所有可用解
            global_optima_list = []
            global_optima_fit_list = []

            # max iteration number should be computed according to specific algorithm
            max_iter = int(max_fes / sp_size)
            num_iter = 0

            gglobal_optima_array = []
            gglobal_r_array = []
            gglobal_optima_array1 = []
            while fes < max_fes:
                for i in range(population_size):
                    gglobal_optima_array.append(population[i, ])
                    gglobal_r_array.append(amps[i])

                # print("fes:", fes)
                # if fes >= 50000:
                #     print("fes:", fes)
                # print("amps:", amps)
                # if amps == 10:
                #     print("fes:", fes)

                # a = np.floor(2 * amps * np.log(dimension + 1)) + 2
                # sp_size_pop = np.floor(1 * amps * np.log(dimension + 1)) + 2
                # sp_size_pop = np.floor(amps ** np.log(dimension + 1)) + 2
                # sp_size_pop = np.floor(amps ** np.log(dimension)) + 2
                # sp_size_pop = np.floor(amps ** np.log(dimension))
                # sp_size_pop = np.where(np.floor(amps ** np.log(dimension)) < 2, 2, np.floor(amps ** np.log(dimension)))
                # sp_size_pop = np.where(np.floor(amps * dimension) < 2, 2, np.floor(amps * dimension))
                # sp_size_pop = np.where(np.floor(amps * 0 + 5) < 2, 2, np.floor(amps * 0 + 5))
                # sp_size_pop = np.floor(2 + np.log(dimension))
                # sp_size_pop = list(map(int, sp_size_pop.tolist()))          # 使用 map() 函数将所有元素转换为整数
                # print(sp_size_pop)

                # Evolve population
                new_population, new_results, temp_population, temp_results, fes = evolve_population(population, results,
                                                                                                    population_size,
                                                                                                    dimension, amps,
                                                                                                    func, fes, lbound,
                                                                                                    ubound, init_amp,
                                                                                                    sp_size_, max_fes,
                                                                                                    sp_size_pop,
                                                                                                    r_rate)

                # # Adaptive sigma for gaussian local search ()
                # exponent = -1 - ((10 / dimension + 3) * (fes / max_fes))
                # sigma = 10 ** exponent
                #
                # # Gaussian local search ()
                # best_fit, best_idv, temp_population, temp_results, new_population, new_results, fes \
                #     = gaussian_ls(min_idx, sigma, best_fit, best_idv, temp_population, temp_results, population_size,
                #                   dimension, lbound, ubound, func, fes, max_fes, new_population, new_results, amps,
                #                   sp_size_)

                # record best results
                min_idx = np.argmax(new_results)  # Max_problem
                now_best_idv = new_population[min_idx, :]
                now_best_fit = new_results[min_idx]
                if now_best_fit >= best_fit:
                    best_idv = now_best_idv
                    best_fit = now_best_fit
                # best_idv = new_population[min_idx, :]
                # best_fit = new_results[min_idx]

                # # Adaptive sigma for gaussian local search ()
                # exponent = -1 - (((10 / dimension + 3) * fes) / max_fes)
                # sigma = 10 ** exponent
                # Gaussian local search ()
                # fes, new_population, new_results, temp_population, temp_results \
                #     = gaussian_ls1(best_fit, best_idv, population_size,
                #                   dimension, lbound, ubound, func, fes, max_fes, amps, init_amp,
                #                   new_population, new_results, temp_population, temp_results)
                # # record best results
                # min_idx = np.argmax(new_results)   # Max_problem
                # now_best_idv = new_population[min_idx, :]
                # now_best_fit = new_results[min_idx]
                # if now_best_fit >= best_fit:
                #     best_idv = now_best_idv
                #     best_fit = now_best_fit

                # # Gaussian local search ()
                # best_fit, best_idv, temp_population, temp_results, new_population, new_results, fes \
                #     = gaussian_ls(min_idx, sigma, best_fit, best_idv, temp_population, temp_results, population_size,
                #                   dimension, lbound, ubound, func, fes, max_fes, new_population, new_results, amps,
                #                   sp_size_)
                # # record best results
                # min_idx = np.argmax(new_results)   # Max_problem
                # now_best_idv = new_population[min_idx, :]
                # now_best_fit = new_results[min_idx]
                # if now_best_fit >= best_fit:
                #     best_idv = now_best_idv
                #     best_fit = now_best_fit

                # Put solutions into archive
                # 按行（垂直方向）合并矩阵  存储候选全局最优解
                concatenated_matrices_v = np.concatenate(temp_population, axis=0)
                concatenated_matrices_f = np.concatenate(temp_results, axis=0)
                # print(len(concatenated_matrices_f))
                if len(global_optima_fit_list) == 0:
                    # concatenated_matrices_v = np.concatenate((concatenated_matrices_v, new_population), axis=0)
                    # concatenated_matrices_f = np.concatenate((concatenated_matrices_f, new_results),
                    #                                          axis=0)
                    pass
                else:
                    # 将 global_optima_list 和 global_optima_fit_list 转换为 numpy 数组
                    temp_global_optima_list = np.array(global_optima_list)
                    temp_global_optima_fit_list = np.array(global_optima_fit_list)
                    global_optima_list = []
                    global_optima_fit_list = []
                    # 合并矩阵
                    concatenated_matrices_v = np.concatenate((concatenated_matrices_v, temp_global_optima_list), axis=0)
                    concatenated_matrices_f = np.concatenate((concatenated_matrices_f, temp_global_optima_fit_list), axis=0)
                    # concatenated_matrices_v = np.concatenate((concatenated_matrices_v, new_population), axis=0)
                    # concatenated_matrices_f = np.concatenate((concatenated_matrices_f, new_results),
                    #                                          axis=0)
                # print(len(concatenated_matrices_f))
                #Archive
                # 计算距离 best_fit 的差值并进行过滤
                threshold = 1e-1
                filtered_indices = [i for i, fit in enumerate(concatenated_matrices_f) if
                                    abs(fit - best_fit) <= threshold]
                # 使用过滤后的索引生成新的矩阵
                concatenated_matrices_v = concatenated_matrices_v[filtered_indices]
                concatenated_matrices_f = concatenated_matrices_f[filtered_indices]
                # print(len(concatenated_matrices_f))

                # 使用 numpy.argsort 获取排序后的索引，注意负号表示降序排序
                sorted_indices = np.argsort(concatenated_matrices_f)[::-1]
                # 使用排序后的索引生成新的矩阵
                concatenated_matrices_v = concatenated_matrices_v[sorted_indices]
                concatenated_matrices_f = concatenated_matrices_f[sorted_indices]

                for i in range(len(concatenated_matrices_f)):  # 去重与半径筛选 ARA

                    is_far_enough = True

                    global_optima_array = np.array(global_optima_list)

                    if global_optima_list:
                        # 计算每个候选解与所有已知最优解之间的距离
                        distances = np.linalg.norm(global_optima_array - concatenated_matrices_v[i],
                                                   axis=1)
                        if np.any(distances <= rho):  # 防止储存太相近的解
                            is_far_enough = False

                    # if len(global_optima_list) <= 300:
                    #     pass
                    # else:
                    #     # 计算每个候选解与所有已知最优解之间的距离
                    #     distances = np.linalg.norm(global_optima_array - concatenated_matrices_v[i],
                    #                                axis=1)
                    #     if np.any(distances <= rho):
                    #         is_far_enough = False

                    if is_far_enough:
                        global_optima_list.append(concatenated_matrices_v[i])
                        global_optima_fit_list.append(concatenated_matrices_f[i])

                    # 保留最优解数量在 archive size 以内
                    if len(global_optima_fit_list) >= 300:
                        break
                # print("fes:", fes)
                # print(len(global_optima_fit_list))
                # print('------------------------------')

                # # if amps < 1e-5:
                # # GB local search (1e-1 to 1e-5)，when amps < 1e-5
                # global_optima_fit_list, global_optima_list, fes, new_population, new_results \
                #     = gaussian_ls(best_fit, best_idv, global_optima_fit_list, global_optima_list, population_size,
                #                   dimension, lbound, ubound, func, fes, max_fes, amps, init_amp,
                #                   new_population, new_results)
                # # record best results
                # min_idx = np.argmax(np.array(global_optima_fit_list))  # Max_problem
                # now_best_idv = global_optima_list[min_idx]
                # now_best_fit = global_optima_fit_list[min_idx]
                # if now_best_fit >= best_fit:
                #     best_idv = now_best_idv
                #     best_fit = now_best_fit

                # Compute the number of global optima found
                for i in range(epsilon_set_size):
                    if not epsilon_flag[i]:
                        temp_global_optima_size, temp_global_optima_idv, temp_global_optima_fit = (
                            compute_global_optima_found(
                                np.array(global_optima_list), np.array(global_optima_fit_list),
                                epsilon_set[i],
                                func))
                        if temp_global_optima_size == global_optima_num:
                            epsilon_flag[i] = True
                        runs_global_num[i][run] = temp_global_optima_size
                        runs_used_fitness[i][run] = fes

                # Record number of found optima with number of fitness evaluations
                if fes >= record_fes[record_count]:
                    for i in range(epsilon_set_size):
                        runs_global_num_vs_fes[i][run][record_count] = runs_global_num[i][run]
                    record_count += 1

                # Restart population
                new_population, new_results, fes = restart_operation(temp_population, temp_results,
                                                                     best_fit, results, new_population, new_results,
                                                                     amps, population_size, max_iter, dimension,
                                                                     num_iter, lbound, ubound, init_amp, func, fes,
                                                                     r_rate, loop_times, rho)

                # new fireworks
                population = new_population
                results = new_results

                # iter and eval num
                num_iter += 1

            # for i in range(len(global_optima_list)):
            #     gglobal_optima_array1.append(global_optima_list[i])
            # gglobal_optima_array1 = np.array(gglobal_optima_array1)
            # # 提取x和y坐标（假设你的数据是二维的）
            # x_coords1 = gglobal_optima_array1[:, 0]
            # y_coords1 = gglobal_optima_array1[:, 1]
            # # 创建一个散点图
            # plt.scatter(x_coords1, y_coords1, c='red', marker='o')
            # # 设置图表标题和标签
            # plt.title('Global Optima Points1')
            # plt.xlabel('X Coordinate')
            # plt.ylabel('Y Coordinate')
            # # 显示图表
            # plt.show()
            # # 在主循环结束后添加
            # gglobal_optima_array = np.array(gglobal_optima_array)
            # gglobal_r_array = np.array(gglobal_r_array)
            # # 提取x和y坐标（假设你的数据是二维的）
            # x_coords = gglobal_optima_array[:, 0]
            # y_coords = gglobal_optima_array[:, 1]
            # # 创建一个散点图
            # plt.scatter(x_coords, y_coords, c='blue', marker='o')
            # # 设置图表标题和标签
            # plt.title('Global Optima Points')
            # plt.xlabel('X Coordinate')
            # plt.ylabel('Y Coordinate')
            # # 显示图表
            # plt.show()

            # # 创建一个绘图
            # fig, ax = plt.subplots()
            # # 遍历每个位置和对应的半径，画出圆
            # # for position, radius in zip(gglobal_optima_array[::5], gglobal_r_array[::5]):
            # for position, radius in zip(gglobal_optima_array[0:1000], gglobal_r_array[0:1000]):
            #     circle = plt.Circle(position, radius, fill=False)
            #     ax.add_patch(circle)
            # # 设置轴的范围
            # ax.set_xlim(np.min(gglobal_optima_array[:, 0]) - np.max(gglobal_r_array),
            #             np.max(gglobal_optima_array[:, 0]) + np.max(gglobal_r_array))
            # ax.set_ylim(np.min(gglobal_optima_array[:, 1]) - np.max(gglobal_r_array),
            #             np.max(gglobal_optima_array[:, 1]) + np.max(gglobal_r_array))
            # # 设置轴的比例
            # ax.set_aspect('equal', adjustable='box')
            # # 显示绘图
            # plt.show()

            # Final computation of global optima found
            for i in range(epsilon_set_size):
                if not epsilon_flag[i]:
                    temp_global_optima_size, temp_global_optima_idv, temp_global_optima_fit = (
                        compute_global_optima_found(
                            np.array(global_optima_list), np.array(global_optima_fit_list),
                            epsilon_set[i],
                            func))  # Search in archive
                    runs_global_num[i][run] = temp_global_optima_size
                    runs_used_fitness[i][run] = max_fes

            # Record number of found optima with number of fitness evaluations
            for i in range(epsilon_set_size):
                runs_global_num_vs_fes[i][run][record_size - 1] = runs_global_num[i][run]

            # print("a")

        # Save results to files
        output_directory = "./Results_gbdide-pop sp set F 0.1 CR 0.8"
        os.makedirs(output_directory, exist_ok=True)

        for i in range(epsilon_set_size):
            epsilon_dir = os.path.join(output_directory, str(i + 1))
            os.makedirs(epsilon_dir, exist_ok=True)

            # Save global optima found and function evaluations
            with open(os.path.join(epsilon_dir, f"Final_Optima_Num_And_FES_Epsilon_{i + 1}_Fun_{fun_to_run[fun]}.txt"),
                      "w") as f:
                for j in range(times_of_run):
                    f.write(f"{runs_global_num[i][j]}\t{runs_used_fitness[i][j]}\n")

            # Save global optima vs function evaluations
            with open(os.path.join(epsilon_dir, f"Optima_Num_VS_FES_Epsilon_{i + 1}_Fun_{fun_to_run[fun]}.txt"),
                      "w") as f:
                for j in range(times_of_run):
                    for k in range(record_size):
                        f.write(f"{runs_global_num_vs_fes[i][j][k]}\t")
                    f.write("\n")

        print(f"Function {fun_to_run[fun]} Finished!")

        print(f"Function pause")


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
