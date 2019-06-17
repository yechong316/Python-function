import random

def single_text_To_two_text(sour_text_path, ques_text_path, ans_text_path, sample=10000):

    with open(sour_text_path, 'r', encoding='utf-8') as f:

        with open(ques_text_path, encoding='utf-8', mode='w') as qus_f:

            with open(ans_text_path, encoding='utf-8', mode='w') as ans_f:

                lines = f.readlines()
                random.shuffle(lines)
                for i in lines[:sample]:

                    line = i.split('\t')
                    ques = line[0]
                    qus_f.write(ques + '\n')


                    ans = line[1]
                    ans_f.write(ans)