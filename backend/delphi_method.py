import warnings

from itertools import chain

import pandas as pd
import numpy as np
import scipy.optimize as optimize

from scipy.special import softmax
from sklearn.metrics import auc
from matplotlib import pyplot as plt

from .utils import gaussian_density, set_argmax, set_argmin


class DelfiMethod(object):
    def __init__(
        self, 
        cases, 
        criteria,
        K,
        S_star,
        R_interval,
        experts_proof,
        W,
        experts_df,
        xs
    ):
        
        self.cases = cases
        self.criteria = criteria
        self.K = K
        self.S_star = S_star
        self.R_interval = R_interval
        self.experts_proof = experts_proof
        self.W = W
        self.experts_df = experts_df
        self.xs = xs
        
        self.solutions = [
        'Надто низький рівень', 
        'Дуже низький рівень', 
        'Низький рівень', 
        'Середній рівень', 
        'Високий рівень', 
        'Дуже високий рівень', 
        'Надто високий рівень'
        ]
        
        self.experts_intervals = {c:{v:self.compute_df_interval(experts_df[c][v]) for v in experts_df[c].keys()} for c in experts_df.keys()}
        self.expected_values = {c:{v:self.compute_expected_value(self.experts_intervals[c][v]) for v in experts_df[c].keys()} for c in experts_df.keys()}
        self.integral_values = {c:{v:self.compute_integral_expert_mark(self.experts_intervals[c][v], self.expected_values[c][v]) for v in experts_df[c].keys()} for c in experts_df.keys()}
        self.integral_gaussian_values = {c:{v:self.compute_integral_gaussian_mark(self.integral_values[c][v]) for v in experts_df[c].keys()} for c in experts_df.keys()}
        self.expert_conf = {c:{v:self.compute_quality_functional(experts_df[c][v], self.integral_gaussian_values[c][v]) for v in experts_df[c].keys()} for c in experts_df.keys()}
        self.median_expert = {c:{v:self.compute_median_expert(self.experts_intervals[c][v]) for v in experts_df[c].keys()} for c in experts_df.keys()}
        self.experts_belive_interval = {c:{v:self.compute_believe_intervals(
            self.median_expert[c][v],
            self.experts_intervals[c][v],
            self.expert_conf[c][v]
        ) for v in experts_df[c].keys()} for c in experts_df.keys()}
        
        self.best_solution = {c:{v:self.compute_best_mark(
            self.median_expert[c][v],
            self.experts_intervals[c][v]
        ) for v in experts_df[c].keys()} for c in experts_df.keys()}
        
    def compute_interval_points(self, mu, v):
        d_min = np.abs(mu - mu*(1-v)*self.K)
        d_min[d_min<0] = 0
        d_max = np.abs(mu + mu*(1-v)*self.K)
        d_max[d_max>1] = 1
        return d_min, d_max
    
    @staticmethod
    def extract_mu_values(df):
        return df.loc[:,list(filter(lambda x: x[1].startswith('Mu') if isinstance(x,tuple) else x.startswith('Mu'), df.columns))]
    
    @staticmethod
    def extract_v_values(df):
        return df.loc[:,list(filter(lambda x: x[1].startswith('V') if isinstance(x,tuple) else x.startswith('V'), df.columns))]
    
    @staticmethod
    def extract_dmin_values(df):
        return df.loc[:,list(filter(lambda x: x[1].startswith('dmin') if isinstance(x,tuple) else x.startswith('dmin'), df.columns))]
    
    @staticmethod
    def extract_dmax_values(df):
        return df.loc[:,list(filter(lambda x: x[1].startswith('dmax') if isinstance(x,tuple) else x.startswith('dmax'), df.columns))]
    
    @staticmethod
    def extract_expert_values(df, expert_id):
        expert_raw = df.iloc[expert_id-1,:]
        expert_raw = [[expert_raw[i],expert_raw[i+1],expert_raw[i+2]] for i in range(0,len(expert_raw),3)]
        return np.array(expert_raw).T
    
    @staticmethod
    def compute_expert_distances(exp_id, d_min, d_max):        
        dist = np.stack([
            np.abs(d_min - d_min[exp_id]),
            np.abs(d_max - d_max[exp_id])
        ], axis=-1).max(axis=-1)
        
        
        dist = dist.mean(axis=1)
        return dist
    
    def compute_best_mark(self, median_expert, df_intervals):
        d_min = DelfiMethod.extract_dmin_values(df_intervals).values
        d_max = DelfiMethod.extract_dmax_values(df_intervals).values
        mu = DelfiMethod.extract_mu_values(df_intervals).values
        
        d_min = d_min[median_expert,:]
        d_max = d_max[median_expert,:]
        mu = mu[median_expert,:]
        
        best_solution  = set_argmax(mu)
        
        if len(best_solution) > 1:
            most_confident_solution = set_argmin(np.abs(d_min - d_max))
            if set(best_solution) & set(most_confident_solution):
                best_solution = list(set(best_solution) & set(most_confident_solution))[0]
                
        else:
            best_solution = best_solution[0]
        
        return best_solution
    
    def compute_believe_intervals(self, median_expert, df_intervals, expert_conf):
        d_min = DelfiMethod.extract_dmin_values(df_intervals).values
        d_max = DelfiMethod.extract_dmax_values(df_intervals).values
        
        dist = DelfiMethod.compute_expert_distances(median_expert, d_min, d_max)
        
        believe_interval = dist * (2 - expert_conf)
                
        return believe_interval < self.R_interval
    
    def compute_median_expert(self, df_intervals):
        d_min = DelfiMethod.extract_dmin_values(df_intervals).values
        d_max = DelfiMethod.extract_dmax_values(df_intervals).values
                
        reduced_experts_distance = []
        for exp_id in range(d_min.shape[0]):
            dist = DelfiMethod.compute_expert_distances(exp_id, d_min, d_max)
            reduced_experts_distance.append(dist.sum())
            
        return np.argmin(reduced_experts_distance)
    
    def compute_quality_functional(self, df, df_integral_gaussian):
        mu = DelfiMethod.extract_mu_values(df).values
        mu_min = np.abs(mu - np.expand_dims(df_integral_gaussian['model_gaussian_down'].values, axis=0))
        mu_max = np.abs(mu - np.expand_dims(df_integral_gaussian['model_gaussian_up'].values, axis=0))
        
        mu = np.stack([mu_min,mu_max], axis=-1)
        mu = np.max(mu, axis=-1)
        dist = mu.mean(axis=1)
                
        return (1 - dist)*np.array(self.experts_proof)
    
    def compute_integral_gaussian_mark(self, df_integral):
        
        k_min = 1 / auc(self.xs, df_integral['model_down'])
        k_max = 1 / auc(self.xs, df_integral['model_up'])
        
        xs = np.array(self.xs)
        
        def func_to_optimize(params):
            mu, d = params
            return np.abs(gaussian_density(xs, k_min, d, mu) - df_integral['model_down']).sum()
        
        result_min = optimize.minimize(
            func_to_optimize, 
            [1, 1],
            bounds=[(-10000, 10000),(1e-12, 10000)])
        
        if not result_min.success:
            warnings.warn("did not converge", DeprecationWarning)
        
        
        def func_to_optimize(params):
            mu, d = params
            return np.abs(gaussian_density(xs, k_max, d, mu) - df_integral['model_up']).sum()
        
        result_max = optimize.minimize(
            func_to_optimize, 
            [1, 1],
            bounds=[(-10000, 10000),(1e-12, 10000)])
        
        if not result_max.success:
            warnings.warn("did not converge", DeprecationWarning)
            
        qmin_gaussian = gaussian_density(xs, k_min, result_min.x[1], result_min.x[0])
        qmax_gaussian = gaussian_density(xs, k_max, result_max.x[1], result_max.x[0])
        
        return pd.DataFrame(
            data = np.array([qmin_gaussian, qmax_gaussian]).T,
            columns = ['model_gaussian_down', 'model_gaussian_up']
        )
        
    
    def compute_integral_expert_mark(self, df_marks, df_expectations):
        mu_values = DelfiMethod.extract_mu_values(df_marks).values
        m_exp_min = df_expectations['dmin_expected'].values
        m_exp_max = df_expectations['dmax_expected'].values
        
        q_min = np.argmin(np.abs(mu_values - m_exp_min), axis=0)
        q_max = np.argmin(np.abs(mu_values - m_exp_max), axis=0)
        
        q_min = [mu_values[q_min[i],i] for i in range(q_min.shape[0])]
        q_max = [mu_values[q_max[i],i] for i in range(q_max.shape[0])]
        
        return pd.DataFrame(
            data=np.array([q_min,q_max]).T,
            columns=['model_down', 'model_up']
        )
    
    def compute_df_interval(self, df):        
        mu_values = DelfiMethod.extract_mu_values(df).values
        v_values = DelfiMethod.extract_v_values(df).values
        
        mu_values = mu_values.T
        v_values = v_values.T
        
        df_interval = []
        for mu, v in zip(mu_values,v_values):
            d_min, d_max = self.compute_interval_points(mu, v)
            df_interval += [d_min, mu, d_max]
            
        np.array(df_interval).T
        df_interval = pd.DataFrame(
            data=np.array(df_interval).T,
            columns=list(chain(*[[
                'dmin_k_{}'.format(i),
                'Mu_k_{}'.format(i),
                'dmax_k_{}'.format(i)] 
                for i in range(1,8)]))
        )
                    
        return df_interval
    
    def compute_expected_value(self, df):
        dmin_values = DelfiMethod.extract_dmin_values(df).values
        dmax_values = DelfiMethod.extract_dmax_values(df).values
        mu_values = DelfiMethod.extract_mu_values(df).values
        
        expected_df = [
            dmin_values.mean(axis=0),
            mu_values.mean(axis=0),
            dmax_values.mean(axis=0)
        ]
        
        expected_df = pd.DataFrame(
            data=np.array(expected_df).T,
            columns=['dmin_expected','mu_expected','dmax_expected']
        )
        
        return expected_df
    
    def plot_point_predictions(self, case, crit):
        df = self.experts_df[case][crit]
        mu_values = DelfiMethod.extract_mu_values(df).values
        for i in range(16):
            plt.plot(mu_values[i], '-o', label='Expert {}'.format(i+1))
              
        plt.title('Точкова оцінка')
        plt.legend()   
        plt.show()
        
    def plot_interval_predictions(self, case, crit, expert_id):
        df = self.experts_intervals[case][crit]
        interval_pred = DelfiMethod.extract_expert_values(df, expert_id)
        plt.plot(interval_pred[0], '-o', label='d_min')
        plt.plot(interval_pred[1], '-o', label='mu')
        plt.plot(interval_pred[2], '-o', label='d_max')
              
        plt.title('Інтревальна оцінка')
        plt.legend()   
        plt.show()
        
    def plot_expected_predictions(self, case, crit):
        df = self.experts_df[case][crit]
        mu_values = DelfiMethod.extract_mu_values(df).values
        for i in range(16):
            plt.plot(mu_values[i], 'o-', label='Expert {}'.format(i+1))
            
        df_expected = self.expected_values[case][crit]
            
        plt.plot(df_expected['dmin_expected'], 'o-', label='dmin_expected', linewidth=7.0)
        plt.plot(df_expected['mu_expected'], 'o-', label='mu_expected', linewidth=7.0)
        plt.plot(df_expected['dmax_expected'], 'o-', label='dmax_expected', linewidth=7.0)
              
        plt.title('Інтервальне математичне середнє інтервальних оцінок')
        plt.legend()   
        plt.show()
        
    def plot_expected_predictions(self, case, crit):
        df = self.experts_df[case][crit]
        mu_values = DelfiMethod.extract_mu_values(df).values
        for i in range(16):
            plt.plot(mu_values[i], 'o-', label='Expert {}'.format(i+1))
            
        df_expected = self.expected_values[case][crit]
            
        plt.plot(df_expected['dmin_expected'], 'o-', label='dmin_expected', linewidth=7.0)
        plt.plot(df_expected['mu_expected'], 'o-', label='mu_expected', linewidth=7.0)
        plt.plot(df_expected['dmax_expected'], 'o-', label='dmax_expected', linewidth=7.0)
              
        plt.title('Інтервальне математичне середнє інтервальних оцінок')
        plt.legend()   
        plt.show()
        
    def plot_integral_predictions(self, case, crit):
        df = self.experts_df[case][crit]
        mu_values = DelfiMethod.extract_mu_values(df).values
        for i in range(16):
            plt.plot(mu_values[i], 'o-', label='Expert {}'.format(i+1), linewidth=0.4)
            
        df_expected = self.expected_values[case][crit]
            
        plt.plot(df_expected['dmin_expected'], 'o-', label='dmin_expected', linewidth=3.0)
        plt.plot(df_expected['mu_expected'], 'o-', label='mu_expected', linewidth=3.0)
        plt.plot(df_expected['dmax_expected'], 'o-', label='dmax_expected', linewidth=3.0)
        
        df_integral = self.integral_values[case][crit]
            
        plt.plot(df_integral['model_down'], 'o-', label='upper_model', linewidth=7.0)
        plt.plot(df_integral['model_up'], 'o-', label='lower_model', linewidth=7.0)
              
        plt.title('Інтервальне математичне середнє інтервальних оцінок')
        plt.legend()   
        plt.show()
        
    def plot_integral_gaussian(self, case, crit):
        df = self.experts_df[case][crit]
        mu_values = DelfiMethod.extract_mu_values(df).values
        for i in range(16):
            plt.plot(mu_values[i], 'o-', label='Expert {}'.format(i+1), linewidth=0.4)
        
        df_integral = self.integral_values[case][crit]
            
        plt.plot(df_integral['model_down'], 'o-', label='upper_model', linewidth=3.0)
        plt.plot(df_integral['model_up'], 'o-', label='lower_model', linewidth=3.0)
        
        df_integral_gaussian = self.integral_gaussian_values[case][crit]
            
        plt.plot(df_integral_gaussian['model_gaussian_up'], 'o-', label='upper_gassian_model', linewidth=7.0)
        plt.plot(df_integral_gaussian['model_gaussian_down'], 'o-', label='lower_gassian_model', linewidth=7.0)
              
        plt.title('Дискретизована інтервальна гауссівська щільність')
        plt.legend()   
        plt.show()
        
        
    def plot_quality_functional(self, case, crit):
        df = self.experts_df[case][crit]
        mu_values = DelfiMethod.extract_mu_values(df).values
        
        expert_values = self.expert_conf[case][crit]
        for i in range(16):
            plt.plot(mu_values[i], 'go-', label='Expert {}'.format(i+1), linewidth=2.0, color=(0.3,0.3,0.5,expert_values[i]))
        
        df_integral_gaussian = self.integral_gaussian_values[case][crit]
            
        plt.plot(df_integral_gaussian['model_gaussian_up'], 'o--', label='upper_gassian_model', linewidth=2.0)
        plt.plot(df_integral_gaussian['model_gaussian_down'], 'o--', label='lower_gassian_model', linewidth=2.0)
              
        plt.title('Оцінки експертів за найвищим та найнижчим функціоналом якості')
        plt.legend()   
        plt.show()
        
    def plot_median_expert(self, case, crit):
        df = self.experts_df[case][crit]
        mu_values = DelfiMethod.extract_mu_values(df).values
        
        median_expert = self.median_expert[case][crit]
        
        df_median = self.experts_intervals[case][crit]
        interval_pred = DelfiMethod.extract_expert_values(df_median, median_expert+1)
        
        for i in range(16):
            if  i == median_expert:
                plt.plot(interval_pred[0], '-o', label='d_min', linewidth=2.0)
                plt.plot(interval_pred[1], '-o', label='Expert median {}'.format(i+1), linewidth=3.0)
                plt.plot(interval_pred[2], '-o', label='d_max', linewidth=2.0)
            else:
                plt.plot(mu_values[i], '-o', label='Expert {}'.format(i+1), linewidth=0.5)
              
        plt.title('Медіана інтервальних оцінок')
        plt.legend()   
        plt.show()
        
    def plot_believed_experts(self, case, crit):
        df = self.experts_df[case][crit]
        mu_values = DelfiMethod.extract_mu_values(df).values
        
        median_expert = self.median_expert[case][crit]
        best_solution = self.best_solution[case][crit]
        
        df_median = self.experts_intervals[case][crit]
        interval_pred = DelfiMethod.extract_expert_values(df_median, median_expert+1)
        
        believed_intervals = self.experts_belive_interval[case][crit]
        believed_experts = np.where(believed_intervals)[0]
        
        print('Оцінки у кластері узгоджені' if believed_intervals.sum() / believed_intervals.shape[0] > self.S_star 
              else 'Оцінки у кластері неузгоджені')
        print('Найкращий вибір {} з ймовірністю єксперта {}'.format(self.solutions[best_solution], mu_values[median_expert, best_solution]))
                
        for i in range(16):
            if  i == median_expert:
                plt.plot(interval_pred[0], 'c-o', label='d_min', linewidth=2.0)
                plt.plot(interval_pred[1], 'm-o', label='Expert median {}'.format(i+1), linewidth=3.0)
                plt.plot(interval_pred[2], 'b-o', label='d_max', linewidth=2.0)
            elif i in believed_experts:
                plt.plot(mu_values[i], 'g-o', label='Expert {}'.format(i+1), linewidth=1.0)
            else:
                plt.plot(mu_values[i], 'k-o', label='Expert {}'.format(i+1), linewidth=1.0)
              
        plt.title('Експертні оцінки, що увійшли до довірчого інтервалу')
        plt.legend()   
        plt.show()