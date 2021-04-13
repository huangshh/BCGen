from utils import *
import sys
import pickle
from nltk.translate.meteor_score import single_meteor_score

class Config:
    def __init__(self):
        self.BATCH_SIZE = 50
        self.EPOCH = 50
        self.MAX_INPUT_SIZE = 400
        self.MAX_OUTPUT_SIZE = 20
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        self.MODEL_SIZE = 400
        self.DICT_SIZE_1 = 30010
        self.DICT_SIZE_2 = 1010
        self.NUM_LAYER = 1
        self.DROPOUT = 0.25
        self.LR = 1e-3

def func(mm):
    ast = mm['ast']
    k = 0
    mask = [True for _ in ast]
    for i in range(len(ast)):
        if ast[k]['num']==ast[i]['num']: k = i
        if ast[i]['num']==0: mask[i] = False
    return get_sbt(k,ast,mask)

def func_seq2seq(mm):
    code = nltk.word_tokenize(mm['code'].strip())
    code = [split_word(x) for x in code]
    code = [x for word in code for x in word]
    return code

def func_codenn(mm):
    code = nltk.word_tokenize(mm['code'].strip())
    code = [split_word(x) for x in code]
    code = [x for word in code for x in word]
    return code

def func_mix(mm):
    sbt = func(mm)
    code = nltk.word_tokenize(mm['code'].strip())
    code = [split_word(x) for x in code]
    code = [x for word in code for x in word]
    return (sbt,code)

def func_addmethod(mm):
    code = nltk.word_tokenize(mm['code'].strip())
    code = [split_word(x) for x in code]
    code = [x for word in code for x in word]
    ast = mm['ast']
    if ast[0]['num']==0: return (([],[]),[])
    num = [0 for _ in ast]
    for i in range(len(ast)-1,-1,-1):
        num[i] = 1
        for j in ast[i]['children']: num[i] += num[j]
    k = 0
    for i in range(len(ast)):
        if ast[i]['num']==ast[0]['num']: 
            k = i
            if num[i]<=200: break
    sbt = get_sbt(k,ast)
    return (sbt,code)

def func_relate(mm, expand_order=1):
    ast = mm['ast']
    code = nltk.word_tokenize(mm['code'].strip())
    code = [split_word(x) for x in code]
    code = [x for word in code for x in word]
    k = 0
    for i in range(len(ast)):
        if ast[k]['num']==ast[i]['num']: k = i
    mask = [True for _ in ast]
    for i in range(len(ast)):
        if ast[i]['num']==0: mask[i] = False
    li = traverse(k,mm['ast'],mask)
    mask = [False for _ in ast]
    for x in li: mask[x] = True
    for _ in range(expand_order):
        names = find_name(ast,mask)
        scopes = []
        for name in names:
            scope = find_name_scope(ast,name)
            colors = []
            for i in names[name]: colors.append(scope[i])
            for i in range(len(scope)):
                if scope[i] in colors: scope[i] = True
                else: scope[i] = False
            scopes.append((name,scope))
        mask_ = find_scope(ast,scopes)
        mask = [x or y for x,y in zip(mask,mask_)]

    num = [0 for _ in ast]
    for i in range(len(ast)-1,-1,-1):
        node = ast[i]
        num[i] = 1 if mask[i] else 0
        for child in node['children']:
            num[i] += num[child]
    for node in ast:
        i = node['id']
        if num[i]>0:
            mask[i] = True
            if node['type'] == 'ForStatement' or node['type'] == 'WhileStatement' or node['type'] == 'IfStatement':
                child = node['children'][0]
                li = traverse(child,ast)
                for elem in li: mask[elem] = True
    num = [0 for _ in ast]
    for i in range(len(ast)-1,-1,-1):
        node = ast[i]
        num[i] = 1 if mask[i] else 0
        for child in node['children']:
            num[i] += num[child]
    
    root = 0
    for i in range(len(ast)):
        if ast[i]['num']==ast[0]['num']:
            root = i
            if num[i]<=200: break

    relate = get_sbt(root,ast,mask)
    return (relate,code)

def get_batch(path,f,config,in_w2i,out_w2i):
    batch_in1, batch_in2, batch_out = [], [], []
    in1_w2i, in2_w2i = in_w2i
    for tu in get_one(f,path):
       in_, out = tu
       in1, in2 = in_
       in1 = [in1_w2i[x] if x in in1_w2i else in1_w2i['<UNK>'] for x in in1]
       in2 = [in2_w2i[x] if x in in2_w2i else in2_w2i['<UNK>'] for x in in2]
       out = [out_w2i[x] if x in out_w2i else out_w2i['<UNK>'] for x in out]
       in1 = in1[:config.MAX_INPUT_SIZE]
       in2 = in2[:config.MAX_INPUT_SIZE]
       out = out[:config.MAX_OUTPUT_SIZE]
       if len(in1)==0: continue
       if len(in2)==0: continue
       if len(out)==0: continue
       batch_in1.append(in1)
       batch_in2.append(in2)
       batch_out.append(out)
       if len(batch_out)>=config.BATCH_SIZE:
           yield (batch_in1, batch_in2), batch_out
           batch_in1, batch_in2, batch_out = [], [], []
    if len(batch_out)>0:
        yield (batch_in1, batch_in2), batch_out

def get_batch_seq2seq(path,f,config,in_w2i,out_w2i):
    batch_in, batch_out = [], []
    for tu in get_one(f,path):
       in_, out = tu
       in_ = [in_w2i[x] if x in in_w2i else in_w2i['<UNK>'] for x in in_]
       out = [out_w2i[x] if x in out_w2i else out_w2i['<UNK>'] for x in out]
       in_ = in_[:config.MAX_INPUT_SIZE]
       out = out[:config.MAX_OUTPUT_SIZE]
       if len(in_)==0: continue
       if len(out)==0: continue
       batch_in.append(in_)
       batch_out.append(out)
       if len(batch_out)>=config.BATCH_SIZE:
           yield batch_in, batch_out
           batch_in, batch_out = [], []
    if len(batch_out)>0:
        yield batch_in, batch_out

def get_batch_codenn(path,f,config,in_w2i,out_w2i):
    batch_in, batch_out = [], []
    for tu in get_one(f,path):
       in_, out = tu
       in_ = [in_w2i[x] if x in in_w2i else in_w2i['<UNK>'] for x in in_]
       out = [out_w2i[x] if x in out_w2i else out_w2i['<UNK>'] for x in out]
       in_ = in_[:config.MAX_INPUT_SIZE]
       out = out[:config.MAX_OUTPUT_SIZE]
       if len(in_)==0: continue
       if len(out)==0: continue
       batch_in.append(in_)
       batch_out.append(out)
       if len(batch_out)>=config.BATCH_SIZE:
           yield batch_in, batch_out
           batch_in, batch_out = [], []
    if len(batch_out)>0:
        yield batch_in, batch_out

def get_batch_mix(path,f,config,in_w2i,out_w2i):
    batch_in1, batch_in2, batch_in3, batch_out = [], [], [], []
    in_w2i, in3_w2i = in_w2i
    in1_w2i, in2_w2i = in_w2i
    for tu in get_one(f,path):
       in_, out = tu
       in_, in3 = in_
       in1,in2 = in_
       in1 = [in1_w2i[x] if x in in1_w2i else in1_w2i['<UNK>'] for x in in1]
       in2 = [in2_w2i[x] if x in in2_w2i else in2_w2i['<UNK>'] for x in in2]
       in3 = [in3_w2i[x] if x in in3_w2i else in3_w2i['<UNK>'] for x in in3]
       out = [out_w2i[x] if x in out_w2i else out_w2i['<UNK>'] for x in out]
       in1 = in1[:config.MAX_INPUT_SIZE]
       in2 = in2[:config.MAX_INPUT_SIZE]
       in3 = in3[:config.MAX_INPUT_SIZE]
       out = out[:config.MAX_OUTPUT_SIZE]
       if len(in1)==0: continue
       if len(in2)==0: continue
       if len(in3)==0: continue
       if len(out)==0: continue
       batch_in1.append(in1)
       batch_in2.append(in2)
       batch_in3.append(in3)
       batch_out.append(out)
       if len(batch_out)>=config.BATCH_SIZE:
           yield ((batch_in1,batch_in2),batch_in3), batch_out
           batch_in1, batch_in2, batch_in3, batch_out = [], [], [], []
    if len(batch_out)>0:
        yield ((batch_in1,batch_in2),batch_in3), batch_out

def get_batch_addmethod(path,f,config,in_w2i,out_w2i):
    return get_batch_mix(path,f,config,in_w2i,out_w2i)


def get_batch_relate(path,f,config,in_w2i,out_w2i):
    return get_batch_mix(path,f,config,in_w2i,out_w2i)

if __name__ == "__main__":
    model_name = sys.argv[1]
    path = sys.argv[2]
    config = Config()
    logger = get_logger('{}/{}_logging.txt'.format(path,model_name))
    type_w2i = pickle.load(open('{}/data/type.w2i'.format(path),'rb'))
    value_w2i = pickle.load(open('{}/data/value.w2i'.format(path),'rb'))
    code_w2i = pickle.load(open('{}/data/code.w2i'.format(path),'rb'))
    nl_w2i = pickle.load(open('{}/data/nl.w2i'.format(path),'rb'))
    type_i2w = pickle.load(open('{}/data/type.i2w'.format(path),'rb'))
    value_i2w = pickle.load(open('{}/data/value.i2w'.format(path),'rb'))
    code_i2w = pickle.load(open('{}/data/code.i2w'.format(path),'rb'))
    nl_i2w = pickle.load(open('{}/data/nl.i2w'.format(path),'rb'))
    if model_name == 'deepcom':
        from deepcom import *
        in_w2i = (type_w2i,value_w2i)
        get_batch = get_batch
        f = func
        model = Model(config)
    elif model_name == 'seq2seq':
        from seq2seq import *
        in_w2i = code_w2i
        get_batch = get_batch_seq2seq
        f = func_seq2seq
        model = Model(config)
    elif model_name == 'codenn':
        from codenn import *
        in_w2i = code_w2i
        get_batch = get_batch_codenn
        f = func_codenn
        model = Model(config)
    elif model_name == 'mix':
        from mix import *
        in_w2i = ((type_w2i,value_w2i),code_w2i)
        get_batch = get_batch_mix
        f = func_mix
        encoder1 = SBTEncoder(config)
        encoder2 = NormalEncoder(config)
        model = Model(config,encoder1,encoder2)
    elif model_name == 'addmethod':
        from mix import *
        in_w2i = ((type_w2i,value_w2i),code_w2i)
        get_batch = get_batch_addmethod
        f = func_addmethod
        encoder1 = SBTEncoder(config)
        encoder2 = NormalEncoder(config)
        model = Model(config,encoder1,encoder2)
    elif model_name == 'relate':
        from mix import *
        in_w2i = ((type_w2i,value_w2i),code_w2i)
        get_batch = get_batch_relate
        f = func_relate
        encoder1 = SBTEncoder(config)
        encoder2 = NormalEncoder(config)
        model = Model(config,encoder1,encoder2)
    
    out_w2i = nl_w2i
    best_bleu = 0.
    for epoch in range(config.EPOCH):
        loss = 0.
        for step,batch in enumerate(get_batch('{}/data/train.json'.format(path),f,config,in_w2i,out_w2i)):
            batch_in, batch_out = batch
            loss += model(batch_in,True,batch_out)
            logger.info('Epoch: {}, Batch: {}, Loss: {}'.format(epoch,step,loss/(step+1)))

        preds = []
        refs = []
        bleu = meteor = rouge = 0.
        for step,batch in enumerate(get_batch('{}/data/valid.json'.format(path),f,config,in_w2i,out_w2i)):
            batch_in, batch_out = batch
            pred = model(batch_in,False)
            preds += pred
            refs += batch_out
            for x,y in zip(batch_out,pred):
                bleu += calc_bleu([x],[y],1)
                meteor += single_meteor_score(' '.join([str(z) for z in x]),' '.join([str(z) for z in y]))
                rouge += myrouge([x],y)
            logger.info('Epoch: {}, Batch: {}, BLEU: {}, METEOR: {}, ROUGE-L: {}'.format(epoch,step,bleu/len(preds),meteor/len(preds),rouge/len(preds)))
        logger.info('Epoch: {}, BLEU: {}, METEOR: {}, ROUGE-L: {}'.format(epoch,bleu/len(preds),meteor/len(preds),rouge/len(preds)))
        
        if bleu>best_bleu:
            best_bleu = bleu
            model.save('{}/model/{}_{}'.format(path,model_name,epoch),model_name)