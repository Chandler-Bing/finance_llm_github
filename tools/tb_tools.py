from  tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
import os

acc_dic = {
    '0.5B':{
        'ceval_0shot_accuracy': 0.424962852897474,
        'ceval_5shot_accuracy':0.4851411589895988,
        'cmmlu_0shot_accuracy':0.3105681229494042,
        'cmmlu_5shot_accuracy':0.4480227939906752,
        'fin_eval_0shot_accuracy':0.3301476976542137,
        'fin_eval_5shot_accuracy':0.43440486533449174,
        'fin_IQ_0shot_accuracy':0.32598624175207075,
        'fin_IQ_5shot_accuracy':0.3943563105433104,
    },
    '1.8B':{
        'ceval_0shot_accuracy':0.5973254086181278,
        'ceval_5shot_accuracy':0.586924219910847,
        'cmmlu_0shot_accuracy':0.5802106717319979,
        'cmmlu_5shot_accuracy': 0.5752892419271283,
        'fin_eval_0shot_accuracy':0.529105125977411,
        'fin_eval_5shot_accuracy':0.5516941789748045,
        'fin_IQ_0shot_accuracy':0.4778885301137161,
        'fin_IQ_5shot_accuracy':0.4818194580934999,
    },
    'baichuan7b+cheat':{
        'ceval_0shot_accuracy':0.6463595839524517,
        'ceval_5shot_accuracy':0.6426448736998515,
        'cmmlu_0shot_accuracy':0.5809013987221551,
        'cmmlu_5shot_accuracy': 0.5922120531859782,
        'fin_eval_0shot_accuracy':0.6238053866203301,
        'fin_eval_5shot_accuracy':0.6159860990443093,
        'fin_IQ_0shot_accuracy': 0.5955355889372456,
        'fin_IQ_5shot_accuracy':0.5871121718377088,
    },
    '7B':{
        'ceval_0shot_accuracy':0.6887072808320951,
        'ceval_5shot_accuracy':0.7288261515601783,
        'cmmlu_0shot_accuracy':0.6162148160939389,
        'cmmlu_5shot_accuracy':  0.7341564496632705,
        'fin_eval_0shot_accuracy':  0.6828844483058211,
        'fin_eval_5shot_accuracy':0.7289313640312771,
        'fin_IQ_0shot_accuracy':  0.6010108100519445,
        'fin_IQ_5shot_accuracy':0.6229116945107399,
    },

}
loss_dic = {
    '1.8B':{
        'ceval_0shot_loss':1.1845976314763436,
        'ceval_5shot_loss':0.9449914568029468,
        'cmmlu_0shot_loss':1.2247135374840445,
        'cmmlu_5shot_loss':0.9984446470994485,
        'fin_eval_0shot_loss':1.3192127681718402,
        'fin_eval_5shot_loss':1.0438519370233192,
        'fin_IQ_0shot_loss':1.4412530835414057,
        'fin_IQ_5shot_loss':1.1646817059524956,
    },
    '0.5B':{
        'ceval_0shot_loss':1.760162295792999,
        'ceval_5shot_loss':1.1641316168691083,
        'cmmlu_0shot_loss':2.1347387600937795,
        'cmmlu_5shot_loss':1.2249548286612681,
        'fin_eval_0shot_loss':2.088610335964415,
        'fin_eval_5shot_loss':1.2623765964504432,
        'fin_IQ_0shot_loss':2.058321758613114,
        'fin_IQ_5shot_loss':1.3098375690531152,
    },
    'baichuan7b+cheat':{
        'ceval_0shot_loss':0.8947398850326198,
        'ceval_5shot_loss': 0.8871106350404192,
        'cmmlu_0shot_loss':1.09487108821141,
        'cmmlu_5shot_loss':1.0854599680478163,
        'fin_eval_0shot_loss':0.9473565480276153,
        'fin_eval_5shot_loss':0.9240184538681335,
        'fin_IQ_0shot_loss':0.9936960970905244,
        'fin_IQ_5shot_loss': 1.0141303646050694,
    },
    '7B':{
        'ceval_0shot_loss': 1.2981064748489484,
        'ceval_5shot_loss': 0.6677683901612925,
        'cmmlu_0shot_loss':1.5974778985915674,
        'cmmlu_5shot_loss': 0.6884574195217156,
        'fin_eval_0shot_loss':1.3674836254295943,
        'fin_eval_5shot_loss': 0.6861964949215726,
        'fin_IQ_0shot_loss':1.4048180278308535,
        'fin_IQ_5shot_loss':  0.8993161131212273,
    },
}



base_dir = '/app/nfs_share_dir/5/boruipeng/tensorboard'

def add_step0_scalar(which):
    input_path = f'{base_dir}/old/{which}'
    child_path = os.listdir(input_path)[0]
    input_path = os.path.join(input_path,child_path)

    output_path = f'{base_dir}/new/{which}'
    output_path = os.path.join(output_path,child_path)

    writer = SummaryWriter(output_path)

    ea = event_accumulator.EventAccumulator(input_path)
    ea.Reload()
    tags = ea.scalars.Keys()
    model = which.split('-')[0]
    loss = loss_dic.get(model)
    acc = acc_dic.get(model)
    for add_tag in loss.keys():
        print(add_tag)
        writer.add_scalar(
            tag=f'eval/{add_tag}',
            scalar_value=loss[add_tag],
            global_step=0,
        )
    for add_tag in acc.keys():
        print(add_tag)
        writer.add_scalar(
            tag=f'eval/{add_tag}',
            scalar_value=acc[add_tag],
            global_step=0,
        )

    for tag in tags:
        scalar_list = ea.scalars.Items(tag)
        for scalar in scalar_list:
            writer.add_scalar(tag,scalar.value,scalar.step,scalar.wall_time)


    writer.close()

def add_scalar_from_text(which):
    input_path = f'{base_dir}/old/ppl/{which}'

    output_path = f'{base_dir}/new/ppl/{which}'

    writer = SummaryWriter(output_path)

    with open(input_path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            step,loss,ppl = line.strip('\n').split('==')
            step,loss,ppl = int(step),float(loss),float(ppl)

            writer.add_scalar(
                tag=f'ppl',
                scalar_value=ppl,
                global_step=step,
            )


    writer.close()

if __name__ == '__main__':
    for key in os.listdir('/app/nfs_share_dir/5/boruipeng/tensorboard/old'):
        if 'ppl' not in key:
            add_step0_scalar(key)
    for which in os.listdir(f'{base_dir}/old/ppl'):
        #print(which)
        add_scalar_from_text(which)
