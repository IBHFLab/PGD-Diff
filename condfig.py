import argparse

def setting_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_log_dir', default='data/output/log',
                        help='Path to log file.')
    
    parser.add_argument('--out_checkpoint_dir', default='data/output/pgd_diff/both_dual_encoder',
                        help='Path to checkpoint file.')
    
    parser.add_argument('--save_top_k', default=400,
                        help='save_top_k for train.')
    
    parser.add_argument('--gpus', default=1,
                        help='gpus for train.')
    
    parser.add_argument('--n_max_epochs', default=200,
                        help='max_epochs for train.')
    
    parser.add_argument('--batch_sizes', default=32,
                        help='batch_sizes for train.')
    
    parser.add_argument('--n_timestep', default=200,
                        help='n_timestep for diffusion model.')
    
    parser.add_argument('--beta_schedule', default='linear',
                        help='beta_schedule for diffusion model.')
    
    parser.add_argument('--beta_start', default=1.e-7,
                        help='beta_start for diffusion model.')
    
    parser.add_argument('--beta_end', default=0.01,
                        help='beta_end for diffusion model.')
    
    parser.add_argument('--temperature', default=0.1,
                        help='temperature for diffusion model.')
    
    parser.add_argument('--learning_rate_struct', default=1e-3,
                        help='learning_rate_struct for diffusion model.')
    
    parser.add_argument('--learning_rate_seq', default=1e-3,
                        help='learning_rate_seq for diffusion model.')
    
    parser.add_argument('--learning_rate_cont', default=1e-3,
                        help='learning_rate_cont for diffusion model.')

    # 关键修改：使用parse_known_args()并只返回args部分
    args, unknown_args = parser.parse_known_args()
    
    return args  # 返回命名空间对象而不是元组