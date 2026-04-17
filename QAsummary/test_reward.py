from reward import em_ner_evidence_reward
import json

question = "该患者最可能的诊断是什么？"
answer = "急性心肌梗死"
article = "患者胸痛3小时，心电图显示ST段抬高，肌钙蛋白增高，提示急性心肌梗死。"

summary_a = "该患者最可能的诊断是急性心肌梗死。"
res_a = em_ner_evidence_reward(question, answer, article, summary_a)
print('A:', json.dumps(res_a, ensure_ascii=False, indent=2))

summary_b = "患者出现持续胸痛并伴有ST段抬高和肌钙蛋白升高，提示急性心肌梗死。"
res_b = em_ner_evidence_reward(question, answer, article, summary_b)
print('B:', json.dumps(res_b, ensure_ascii=False, indent=2))
