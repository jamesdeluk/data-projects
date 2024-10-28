import random
from time import time

print('\nWelcome!\n')

rnd=1

def report_score_and_time(score, r, session_time):
    percent = int((score / r) * 100)
    print(f'Score: {score} / {r} ({percent}%)')
    print(f'Time: {int((session_time / 60) * 10) / 10} minutes')
    print(f'Avg {int((session_time / r) * 10) / 10} seconds per answer')
    if score > 0:
        print(f'Avg {int((session_time / score) * 10) / 10} seconds per correct answer')

def input_with_default(prompt, default, data_type=str, range_split=False):
    user_input = input(prompt)
    if user_input == '':
        return default
    if data_type == int and range_split:
        f1, f2 = map(int, user_input.split('-'))
        return f1, f2
    if data_type == float and not range_split:
        return (100 - float(user_input)) / 100, (100 + float(user_input)) / 100
    return data_type(user_input)

while True:
    print('Select mode:')
    print('[1] Addition')
    print('[2] Subtraction')
    print('[3] Multiplication')
    print('[4] Division')
    print('[5] Divisibility')
    print('[6] Percentage')
    print('[7] Shuffle (default)')
    print('\n')
    choice=input('Choice: ')
    print('\n')
    score=0
    total_time=0
    start=0.0

    print(f'Round {rnd}\n')

    if choice == '1': # Addition
        r = input_with_default('Number of rounds [default 10]:\t', 10, int)
        f1, f2 = input_with_default('First number [default 1-9999]:\t', (1, 9999), int, range_split=True)
        s1, s2 = input_with_default('Second number [default 1-9999]:\t', (1, 9999), int, range_split=True)
        t1, t2 = input_with_default('Tolerance % [default ±0]:\t', (1, 1), float)
        start=time()
        print('\n')
        for i in range(r):
            a=random.randint(f1,f2)
            b=random.randint(s1,s2)
            c=a+b
            ans=input(f'{i+1}: {a} + {b} = ')
            if ans == '':
                print(f'Incorrect: {a} + {b} = {c}\n')
            elif ans == 'q':
                break
            elif ans == c:
                print('Perfect!\n')
                score+=1
            elif float(ans)*t1 <= c <= float(ans)*t2:
                print('Correct\n')
                score+=1
            else:
                print(f'Incorrect: {a} + {b} = {c}\n')
        end=time()
        session_time=end-start
        total_time+=session_time
        report_score_and_time(score, r, session_time)
    elif choice == '2': # Subtraction
        r = input_with_default('Number of rounds [default 10]:\t', 10, int)
        f1, f2 = input_with_default('First number [default 1-9999]:\t', (1, 9999), int, range_split=True)
        s1, s2 = input_with_default('Second number [default 1-9999]:\t', (1, 9999), int, range_split=True)
        t1, t2 = input_with_default('Tolerance % [default ±0]:\t', (1, 1), float)
        start=time()
        print('\n')
        for i in range(r):
            a=random.randint(f1,f2)
            b=random.randint(s1,s2)
            c=a-b
            ans=input(f'{i+1}: {a} - {b} = ')
            if ans == '':
                print(f'Incorrect: {a} - {b} = {c}\n')
            elif ans == 'q':
                break
            elif ans == c:
                print('Perfect!\n')
                score+=1
            elif float(ans)*t1 <= c <= float(ans)*t2:
                print('Correct\n')
                score+=1
            else:
                print(f'Incorrect: {a} - {b} = {c}\n')
        end=time()
        session_time=end-start
        total_time+=session_time
        report_score_and_time(score, r, session_time)
    elif choice == '3': # Multiplication
        r = input_with_default('Number of rounds [default 10]:\t', 10, int)
        f1, f2 = input_with_default('First number [default 3-99]:\t', (3, 99), int, range_split=True)
        s1, s2 = input_with_default('Second number [default 3-99]:\t', (3, 99), int, range_split=True)
        t1, t2 = input_with_default('Tolerance % [default ±0]:\t', (1, 1), float)
        start=time()
        print('\n')
        for i in range(r):
            a=random.randint(f1,f2)
            b=random.randint(s1,s2)
            c=a*b
            ans=input(f'{i+1}: {a} * {b} = ')
            if ans == '':
                print(f'Incorrect: {a} * {b} = {c}\n')
            elif ans == 'q':
                break
            elif float(ans) == float(c):
                print('Perfect!\n')
                score+=1
            elif float(ans)*t1 <= c <= float(ans)*t2:
                print('Correct\n')
                score+=1
            else:
                print(f'Incorrect: {a} * {b} = {c}\n')
        end=time()
        session_time=end-start
        total_time+=session_time
        report_score_and_time(score, r, session_time)
    elif choice == '4': # Division
        r = input_with_default('Number of rounds [default 10]:\t', 10, int)
        f1, f2 = input_with_default('First number [default 9-999]:\t', (9, 999), int, range_split=True)
        s1, s2 = input_with_default('Second number [default 2-19]:\t', (2, 19), int, range_split=True)
        t1, t2 = input_with_default('Tolerance % [default ±1]:\t', (0.99, 1.01), float)
        start=time()
        print('\n')
        for i in range(r):
            a=random.randint(f1,f2)
            b=random.randint(s1,s2)
            c=a/b
            ans=input(f'{i+1}: {a} / {b} = ')
            if ans == '':
                print(f'Incorrect: {a} / {b} = {c}\n')
            elif ans == 'q':
                break
            elif float(ans) == float(c):
                print('Perfect!\n')
                score+=1
            elif float(ans)*t1 <= c <= float(ans)*t2:
                print(f'Correct [{a/b}]\n')
                score+=1
            else:
                print(f'Incorrect: {a} / {b} = {c}\n')
        end=time()
        session_time=end-start
        total_time+=session_time
        report_score_and_time(score, r, session_time)
    elif choice == '5': # Divisibility
        r = input_with_default('Number of rounds [default 10]:\t', 10, int)
        f1, f2 = input_with_default('Dividend range [default 99-999]:\t', (99, 999), int, range_split=True)
        s1, s2 = input_with_default('Divisor range [default 2-9]:\t', (2, 9), int, range_split=True)
        start=time()
        print('\n')
        for i in range(r):
            a=random.randint(f1,f2)
            b=random.randint(s1,s2)
            c=a/b
            ans=input(f'{i+1}: Is {a} perfectly divisible by {b}? [y/n]: ')
            if ans == '':
                print(f'{a} / {b} = {c}\n')
            elif ans == 'q':
                break
            elif a%b == 0.0 and ans == 'y':
                print(f'Correct: {a} / {b} = {int(c)} r {a%b}\n')
                score+=1
            elif a%b == 0.0 and ans == 'n':
                print(f'Incorrect: {a} / {b} = {int(c)} r {a%b}\n')
            elif a%b != 0.0 and ans == 'y':
                print(f'Incorrect: {a} / {b} = {int(c)} r {a%b}\n')
            elif a%b != 0.0 and ans == 'n':
                print(f'Correct: {a} / {b} = {int(c)} r {a%b}\n')
                score+=1
            else:
                print(f'Incorrect: {a} / {b} = {int(c)} r {a%b}\n')
        end=time()
        session_time=end-start
        total_time+=session_time
        report_score_and_time(score, r, session_time)
    elif choice == '6': # Percentage
        r = input_with_default('Number of rounds [default 10]:\t', 10, int)
        f1, f2 = input_with_default('First number [default 3-99]:\t', (3, 99), int, range_split=True)
        s1, s2 = input_with_default('Second number [default 3-99]:\t', (3, 99), int, range_split=True)
        t1, t2 = input_with_default('Tolerance % [default ±5]:\t', (5, 5), float)
        start=time()
        print('\n')
        for i in range(r):
            a=random.randint(f1,f2)
            b=random.randint(s1,s2)
            if a<b:
                a,b=b,a
            c=round(b/a*100,1)
            ans=input(f'{i+1}: What percentage of {a} is {b}? (1dp) ')
            if ans == '':
                print(f'Incorrect: {b} is {c}% of {a}\n')
            elif ans == 'q':
                break
            elif float(ans) == float(c):
                print('Perfect!\n')
                score+=1
            elif float(ans)-t1 <= c <= float(ans)+t2:
                print(f'Correct [{c}%]\n')
                score+=1
            else:
                print(f'Incorrect: {b} is {c}% of {a}\n')
        end=time()
        session_time=end-start
        total_time+=session_time
        report_score_and_time(score, r, session_time)
    elif choice == 'q':
        print('End\n')
    else: # Shuffle
        start=time()
        r=25
        for i in range(r):
            func=random.randint(0,4)
            if func==0: # Addition
                a=random.randint(1,9999)
                b=random.randint(1,9999)
                c=a+b
                ans=input(f'{i+1}: {a} + {b} = ')
                if ans == '':
                    print(f'Incorrect: {a} + {b} = {c}\n')
                elif ans == 'q':
                    break
                elif int(ans) == c:
                    print('Perfect!\n')
                    score+=1
                else:
                    print(f'Incorrect: {a} + {b} = {c}\n')
            elif func==1: # Subtraction
                a=random.randint(1,9999)
                b=random.randint(1,9999)
                c=a-b
                ans=input(f'{i+1}: {a} - {b} = ')
                if ans == '':
                    print(f'Incorrect: {a} - {b} = {c}\n')
                elif ans == 'q':
                    break
                elif int(ans) == c:
                    print('Perfect!\n')
                    score+=1
                else:
                    print(f'Incorrect: {a} - {b} = {c}\n')
            elif func==2: # Multiplication
                a=random.randint(3,99)
                b=random.randint(3,99)
                c=a*b
                ans=input(f'{i+1}: {a} * {b} = ')
                if ans == '':
                    print(f'Incorrect: {a} * {b} = {c}\n')
                elif ans == 'q':
                    break
                elif float(ans) == float(c):
                    print('Perfect!\n')
                    score+=1
                else:
                    print(f'Incorrect: {a} * {b} = {c}\n')
            elif func==3: # Division
                a=random.randint(99,999)
                b=random.randint(2,19)
                c=a/b
                ans=input(f'{i+1}: {a} / {b} = ')
                if ans == '':
                    print(f'Incorrect: {a} / {b} = {c}\n')
                elif ans == 'q':
                    break
                elif float(ans) == float(c):
                    print('Perfect!\n')
                    score+=1
                elif float(ans)*0.99 <= c <= float(ans)*1.01:
                    print(f'Within 1% [{a/b}]\n')
                    score+=1
                elif float(ans)-1 <= c <= float(ans)+1:
                    print(f'Within ±1 [{a/b}]\n')
                    score+=1
                else:
                    print(f'Incorrect: {a} / {b} = {c}\n')
            elif func==4: # Percentage
                a=random.randint(3,99)
                b=random.randint(3,99)
                if a<b:
                    a,b=b,a
                c=round(b/a*100,1)
                ans=input(f'{i+1}: What percentage of {a} is {b}? (within 2.5%) ')
                if ans == '':
                    print(f'Incorrect: {b} is {c}% of {a}\n')
                elif ans == 'q':
                    break
                elif float(ans) == float(c):
                    print('Perfect!\n')
                    score+=1
                elif float(ans)-2.5 <= c <= float(ans)+2.5:
                    print(f'Correct [{c}%]\n')
                    score+=1
                else:
                    print(f'Incorrect: {b} is {c}% of {a}\n')
        end=time()
        session_time=end-start
        total_time+=session_time
        report_score_and_time(score, r, session_time)
    play=input("Play again? [y/n]: ")
    if play == 'n':
        break
    else:
        rnd+=1
        print('\n')
        continue

print('\nThanks!')
print(f'Total time: {int(((total_time)/60)*10)/10} minutes\n')