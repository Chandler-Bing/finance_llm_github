from .common import CommonExam

class FinEval(CommonExam):
    def __init__(self,shot,data_path='./exams/fin_eval'):
        categories = {
            "accounting": "会计",
            "advanced_financial_accounting": "高级财务会计",
            "auditing": "审计学",
            "corporate_strategy_and_risk_management": "公司战略与风险管理",
            "cost_accounting": "成本会计学",
            "economic_law": "经济法",
            "financial_management": "财务管理学",
            "intermediate_financial_accounting": "中级财务会计",
            "management_accounting": "管理会计学",
            "tax_law": "税法",
            "banking_practitioner_qualification_certificate": "银行从业资格证",
            "certified_management_accountant": "管理会计师",
            "certified_practising_accountant": "注册会计师",
            "china_actuary": "中国精算师",
            "fund_qualification_certificate": "基金从业资格证",
            "futures_practitioner_qualification_certificate": "期货从业资格证",
            "securities_practitioner_qualification_certificate": "证券从业资格证",
            "econometrics": "计量经济学",
            "international_economics": "国际经济学",
            "macroeconomics": "宏观经济学",
            "microeconomics": "微观经济学",
            "political_economy": "政治经济学",
            "public_finance": "财政学",
            "statistics": "统计学",
            "central_banking": "中央银行学",
            "commercial_bank_finance": "商业银行金融学",
            "corporate_finance": "公司金融学",
            "finance": "金融学",
            "financial_engineering": "金融工程学",
            "financial_markets": "金融市场学",
            "insurance": "保险学",
            "international_finance": "国际金融学",
            "investments": "投资学",
            "monetary_finance": "货币金融学"
        }
        super(FinEval, self).__init__(data_path=data_path,shot=shot,cot=True,categories=categories)


class FinEval2(CommonExam):
    def __init__(self,shot,data_path='./exams/fin_eval2'):
        categories = {
            "accounting": "会计",
            "advanced_financial_accounting": "高级财务会计",
            "auditing": "审计学",
            "corporate_strategy_and_risk_management": "公司战略与风险管理",
            "cost_accounting": "成本会计学",
            "economic_law": "经济法",
            "financial_management": "财务管理学",
            "intermediate_financial_accounting": "中级财务会计",
            "management_accounting": "管理会计学",
            "tax_law": "税法",
            "banking_practitioner_qualification_certificate": "银行从业资格证",
            "certified_management_accountant": "管理会计师",
            "certified_practising_accountant": "注册会计师",
            "china_actuary": "中国精算师",
            "fund_qualification_certificate": "基金从业资格证",
            "futures_practitioner_qualification_certificate": "期货从业资格证",
            "securities_practitioner_qualification_certificate": "证券从业资格证",
            "econometrics": "计量经济学",
            "international_economics": "国际经济学",
            "macroeconomics": "宏观经济学",
            "microeconomics": "微观经济学",
            "political_economy": "政治经济学",
            "public_finance": "财政学",
            "statistics": "统计学",
            "central_banking": "中央银行学",
            "commercial_bank_finance": "商业银行金融学",
            "corporate_finance": "公司金融学",
            "finance": "金融学",
            "financial_engineering": "金融工程学",
            "financial_markets": "金融市场学",
            "insurance": "保险学",
            "international_finance": "国际金融学",
            "investments": "投资学",
            "monetary_finance": "货币金融学"
        }
        super(FinEval2, self).__init__(data_path=data_path,shot=shot,cot=True,categories=categories)
if __name__ == '__main__':
    from tqdm import tqdm
    import time
    import json
    a = FinEval2(shot=0)
    for data in tqdm(a,total=len(a)):
        text = data['prompt']
        answer = data['answer']
        print(json.dumps({'text':text,'answer':answer,'tag':f'fin_eval-{data["subject"]}'},ensure_ascii=False))
        #time.sleep(1)