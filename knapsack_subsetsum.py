import pandas as pd

def min_subset_sum_partition(nums, precision):
    scale = 10 ** precision
    scaled = [round(x * scale) for x in nums]
    n = len(scaled)
    total = sum(scaled)
    half = total // 2

    # prefix[i] = reachable subset sums
    prefix = [1] * (n + 1)
    for i, w in enumerate(scaled, start=1):
        prefix[i] = prefix[i - 1] | (prefix[i - 1] << w) if w else prefix[i - 1]

    # best sum for side A
    s1 = (prefix[n] & ((1 << (half + 1)) - 1)).bit_length() - 1

    # backtrack
    A, s = [], s1
    for i in range(n, 0, -1):
        if (prefix[i - 1] >> s) & 1:          # reachable without item i -> B
            continue
        A.append(i - 1)                       # item i was necessary   -> A
        s -= scaled[i - 1]
    A = sorted(A)
    in_A = set(A)
    B = [i for i in range(n) if i not in in_A]
    return A, B, (total - 2 * s1) / scale


data = pd.read_csv(r"number_set_100.csv")
arr = data['Number'].tolist()

A, B, diff = min_subset_sum_partition(arr, precision=9)
sum_A = sum(arr[i] for i in A)
sum_B = sum(arr[i] for i in B)

print(f"Reciprocal of the difference: {1/abs(sum_A - sum_B)}")

print(f"Minimum subset sum difference (scaled): {diff}")
print(f"Subset A: {len(A)} elements, true sum = {sum_A:.9f}")
print(f"Subset B: {len(B)} elements, true sum = {sum_B:.9f}")
print(f"True float difference of this partition: {abs(sum_A - sum_B):.2e}")
print(f"A values : {[round(arr[i], 6) for i in A]}")