import csv
import random

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator


class SpambaseEvaluator(SimpleIndividualEvaluator):
    def __init__(self):
        super().__init__()
        with open("spambase.csv") as spambase:
            spamReader = csv.reader(spambase)
            self.spam = list(list(float(elem) for elem in row) for row in spamReader)

    def evaluate_individual(self, individual):
        # Randomly sample 400 mails in the spam database
        spam_samp = random.sample(self.spam, 450)
        # Evaluate the sum of correctly identified mail as spam
        results = []
        for mail in spam_samp:
            results.append(bool(
                individual.execute(v0=mail[0], v1=mail[1], v2=mail[2], v3=mail[3], v4=mail[4], v5=mail[5], v6=mail[6],
                                   v7=mail[7], v8=mail[8], v9=mail[9], v10=mail[10], v11=mail[11], v12=mail[12],
                                   v13=mail[13], v14=mail[14], v15=mail[15], v16=mail[16], v17=mail[17], v18=mail[18],
                                   v19=mail[19], v20=mail[20], v21=mail[21], v22=mail[22], v23=mail[23], v24=mail[24],
                                   v25=mail[25], v26=mail[26], v27=mail[27], v28=mail[28], v29=mail[29], v30=mail[30],
                                   v31=mail[31], v32=mail[32], v33=mail[33], v34=mail[34], v35=mail[35], v36=mail[36],
                                   v37=mail[37], v38=mail[38], v39=mail[39], v40=mail[40], v41=mail[41], v42=mail[42],
                                   v43=mail[43], v44=mail[44], v45=mail[45], v46=mail[46], v47=mail[47], v48=mail[48],
                                   v49=mail[49], v50=mail[50], v51=mail[51], v52=mail[52], v53=mail[53], v54=mail[54],
                                   v55=mail[55], v56=mail[56])) is bool(mail[57]))
        return sum(results)
