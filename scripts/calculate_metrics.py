import editdistance
import pandas as pd
import os

def get_cer(word1, word2):
  if type(word1) is not str or type(word2) is not str:
    return 1

  gt_cer, pd_cer = list(word1), list(word2)
  cer_distance = editdistance.eval(gt_cer, pd_cer)
  return cer_distance / max(len(gt_cer), 1)

def get_wer(word1, word2):
  if type(word1) is not str or type(word2) is not str:
    return 1

  gt_wer, pd_wer = word1.split(), word2.split()
  wer_distance = editdistance.eval(gt_wer, pd_wer)
  return wer_distance / max(len(gt_wer), 1)

models = ['byt5', 'bart', 'sabia', 'gervasio', 'symspell', 'norvig', 'ngram']

ocrs = ['bluche', 'flor', 'puigcerver', 'pero','ltu_main','ltu_ensemble','litis','demokritos', 'azure']

ground_truth = pd.read_csv('experiments/test_ground_truth.csv')

for model in models:
  for ocr in ocrs:
    text = ''
    metrics = pd.DataFrame([], columns=['model', 'ocr, cer_baseline', 'wer_baseline', 'cer_sc', 'wer_sc'])
    predictions = pd.read_csv(f'experiments/test_data/{ocr}.csv')
    corrections = pd.read_csv(f'experiments/third/corrections/{model}/{ocr}.csv')
    for index in range(len(predictions)):
      baseline = predictions['prediction'][index].strip() if isinstance(predictions['prediction'][index], str) else ""
      target = ground_truth['ground_truth'][index].strip() if isinstance(ground_truth['ground_truth'][index], str) else ""
      correction = corrections['prediction'][index].strip() if isinstance(corrections['prediction'][index], str) else ""

      cer_baseline = get_cer(target, baseline) if baseline != "SKIP" else 0
      cer_sc = get_cer(target, correction) if correction != "SKIP" else 0
      wer_baseline = get_wer(target, baseline) if baseline != "SKIP" else 0
      wer_sc = get_wer(target, correction) if correction != "SKIP" else 0

      metric = pd.DataFrame([{'model': model, 'ocr': ocr, 'cer_baseline': cer_baseline, 'wer_baseline': wer_baseline, 'cer_sc': cer_sc, 'wer_sc': wer_sc}])

      metrics = pd.concat([metrics, metric], ignore_index=True)

    total = len(predictions)

    if model == 'azure':
        total = len(predictions[predictions['prediction'] != 'SKIP'])

    cer_baseline_mean = metrics['cer_baseline'].sum() / total
    wer_baseline_mean = metrics['wer_baseline'].sum() / total

    cer_sc_mean = metrics['cer_sc'].sum() / total
    wer_sc_mean = metrics['wer_sc'].sum() / total

    text += f'''CER - baseline : {(cer_baseline_mean * 100):.2f}%
WER - baseline : {(wer_baseline_mean * 100):.2f}%
CER - corrector: {(cer_sc_mean * 100):.2f}%
WER - corrector: {(wer_sc_mean * 100):.2f}%
    '''

    if not os.path.exists(f'experiments/third/metrics/{model}'):
        os.makedirs(f'experiments/third/metrics/{model}')

    with open(f'experiments/third/metrics/{model}/{ocr}.txt', 'w') as f:
        f.write(text)