import math

def calculate_error_exponent(p, R):
    """
    计算 BSC 信道的 Gallager 随机编码误差指数 E_r(R)
    通过网格搜索法在 [0, 1] 范围内寻找最优的 rho
    """
    best_Er = -1
    steps = 1000
    
    for i in range(steps + 1):
        rho = i / steps
        # 计算 E0(rho)
        term1 = p ** (1 / (1 + rho))
        term2 = (1 - p) ** (1 / (1 + rho))
        E0_rho = rho - (1 + rho) * math.log2(term1 + term2)
        
        # 计算 Er(rho, R)
        Er_rho = E0_rho - rho * R
        if Er_rho > best_Er:
            best_Er = Er_rho
            
    return best_Er

def main():
    # 设定参数
    p = 0.1  # 二进制对称信道的翻转概率 10%
    # 信道容量 C = 1 - H(p) = 1 - (-0.1*log2(0.1) - 0.9*log2(0.9)) ≈ 0.531 bit/use
    R = 0.3  # 设定传输码率 R (必须满足 R < C 才能使错误概率收敛到0)
    
    # 1. 计算误差指数 E_r(R)
    Er = calculate_error_exponent(p, R)
    print(f"信道翻转概率: p = {p}")
    print(f"目标传输码率: R = {R}")
    print(f"计算得出信道误差指数 (Reliability): E_r(R) ≈ {Er:.6f}\n")
    
    # 2. 打印 编码长度 N -> 失败概率上限 P_e 的关系
    print(f"{'编码长度 (N)':<15} -> {'传输失败概率上限 (P_e)':<20}")
    print("-" * 45)
    
    # 选取一系列不断增长的编码长度
    lengths =[10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    for N in lengths:
        # P_e <= 2^(-N * E_r(R))
        Pe_bound = 2 ** (-N * Er)
        print(f"N = {N:<11} -> P_e <= {Pe_bound:.4e}")

if __name__ == "__main__":
    main()