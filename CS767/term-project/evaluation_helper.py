import rouge

'this module evaluates summaries using rouge: ref: https://pypi.org/project/py-rouge/'
'input: list of hypotheses and list of references optional evaluate individual inputs'
'return: formated string of scores.'


def evaluate_rouge_score(hypotheses, references):

    evaluator = rouge.Rouge(metrics=['rouge-n','rouge-l','rouge-w'],
            max_n = 4,
            limit_length = True,
            length_limit = 100,
            length_limit_type = 'words',
            apply_avg = True, 
            apply_best = True, 
            alpha = 0.5,
            weight_factor = 1.2, 
            stemming=True)

    scores = evaluator.get_scores(hypotheses, references)
    results_str = ''
    for metric, results in sorted(scores.items(), key=lambda x:x[0]):
        results_str += '\t{}:\tPrecision: {:5.2f}\tRecall: {:5.2f}\tF1-Score: {:5.2f}\n'.format(metric, 
                100.0*results['p'], 
                100.0*results['r'],
                100.0*results['f'])
        
    return results_str



