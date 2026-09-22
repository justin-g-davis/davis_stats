def quartile_practice():
    import random
    print("Quartile practice (5 questions)")
    print("Odd n: exclude the median when finding Q1/Q3.\n")
    for question in range(1, 6):
        n = random.randint(4, 10)
        data = sorted(random.sample(range(1, 100), n))
        # Easy numbers: each value equals the previous, or differs by 1 or 2
        start = random.randint(1, 80)
        data = [start]
        for _ in range(n - 1):
            nxt = data[-1] + random.choice([0, 1, 2])
            data.append(min(nxt, 99))
        data = sorted(data)
        which = random.choice(["Q1", "Q2", "Q3"])
        # Q2 = median of all data
        mid = n // 2
        if n % 2 == 1:
            q2 = data[mid]
            lower = data[:mid]
            upper = data[mid + 1 :]
        else:
            q2 = (data[mid - 1] + data[mid]) / 2
            lower = data[:mid]
            upper = data[mid:]
        # Q1 = median of lower half
        m = len(lower) // 2
        if len(lower) % 2 == 1:
            q1 = lower[m]
        else:
            q1 = (lower[m - 1] + lower[m]) / 2
        # Q3 = median of upper half
        m = len(upper) // 2
        if len(upper) % 2 == 1:
            q3 = upper[m]
        else:
            q3 = (upper[m - 1] + upper[m]) / 2
        correct = {"Q1": q1, "Q2": q2, "Q3": q3}[which]
        print(f"Question {question} of 5")
        print(f"Data ({n} values): {data}")
        ans = input(f"What is {which}? ").strip()
        try:
            student = float(ans)
        except ValueError:
            show = int(correct) if float(correct).is_integer() else correct
            print(f"Incorrect. Correct {which} = {show}\n")
            continue
        if abs(student - correct) < 1e-9:
            print("Correct!\n")
        else:
            show = int(correct) if float(correct).is_integer() else correct
            print(f"Incorrect. Correct {which} = {show}\n")
    print("Done.")
