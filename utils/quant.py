P = 8888


def quant(raw_ratios, limit = 1000000):
    num_list = []
    n_list = []

    for M in raw_ratios:
        if M == 0:
            num_list.append(0)
            n_list.append(0)
            continue

        best_n = 0
        best_Mo = 0
        min_dif = float('inf')  # 记录最小的 dif 绝对值

        for n in range(10, 27):
            result = M * P
            Mo = int(round(2 ** n * M))
            approx_result = (Mo * P) >> n
            dif = M - Mo * 2 ** (-n)
            # print("M=%f, n=%d, Mo=%d, error=%f,dif=%.11f" % \
            #       (M, n, Mo, result - approx_result, dif))

            # 满足原脚本的筛选条件
            if result - approx_result < 5 and abs(dif) < 0.00001 and Mo < limit:
                # 寻找 dif 绝对值最小的作为最优解
                if abs(dif) < min_dif:
                    min_dif = abs(dif)
                    best_n = n
                    best_Mo = Mo

        num_list.append(best_Mo)
        n_list.append(best_n)

    return num_list, n_list