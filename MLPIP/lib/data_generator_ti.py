import numpy as np
import os
import pdb


class TieredGenerator():
    def __init__(self, seed=0, root='data/tiered-imagenet', config={}):
        np.random.seed(seed)
        xtr = np.load(os.path.join(root, 'imtiered_train.npy'), encoding='latin1')
        xte = np.load(os.path.join(root, 'imtiered_test.npy'), encoding='latin1')
        xval = np.load(os.path.join(root, 'imtiered_val.npy'), encoding='latin1')
        # encoding is for python3

        np.random.shuffle(xtr)
        np.random.shuffle(xte)
        np.random.shuffle(xval)

        nxtr = [[2]*50,[4]*50,[6]*50,[8]*40,[10]*30,[14]*30,[18]*20,[24]*17,[30]*15,[36]*13,[42]*10,[48]*9,[60]*7,[80]*5,[100]*3,[130]*2]
        nxval =[[2]*12,[4]*12,[6]*12,[8]*10,[10]*10,[14]*8,[18]*8,[24]*4,[30]*4,[36]*4,[42]*3,[48]*3,[60]*3,[80]*2,[100]*1,[130]*1]
        nxte = [[2]*20,[4]*20,[6]*20,[8]*20,[10]*10,[14]*10,[18]*10,[24]*10,[30]*8,[36]*8,[42]*6,[48]*6,[60]*4,[80]*4,[100]*2,[130]*2]
        ntr = np.concatenate(nxtr)
        nte = np.concatenate(nxte)
        nval = np.concatenate(nxval)

        xtr_, xval_, xte_ = [],[],[]
        for k in range(len(xtr)):
            xtr_.append(np.reshape(xtr[k][:ntr[k]], [-1,84,84,3]))
        for k in range(len(xte)):
            xte_.append(np.reshape(xte[k][:nte[k]], [-1,84,84,3]))
        for k in range(len(xval)):
            xval_.append(np.reshape(xval[k][:nval[k]], [-1,84,84,3]))
            
        self.train_data = np.array(xtr_)
        self.test_data = np.array(xte_)
        self.val_data = np.array(xval_)
        print ('tiered imagenet loaded')

#    def _data_queue(self, mode, meta_batch_size, n_classes):
#        data = self.train_data if mode == 'train' else self.test_data
#        anyind = [(np.random.choice(10,1)[0] + 1) * 2 for k in range(n_classes)]
#        mxtr, mxte, mytr, myte = [], [], [], []
#        for _ in range(meta_batch_size):
#            tasks = np.arange(len(data))
#            np.random.shuffle(tasks)
#            tasks = tasks[:n_classes]
#
#            np.random.shuffle(anyind)
#            xtr, xte, ytr, yte = [], [], [], []
#            for n in range(n_classes):
#                tmp = data[tasks[n]]
#                np.random.shuffle(tmp)
#                tmp = tmp[:anyind[n]]
#                tr, te = np.split(tmp, 2)
#                xtr.append(tr)
#                xte.append(te)
#                ytr.append([n]*len(tr))
#                yte.append([n]*len(te))
#            
#            ytr = to1hot(np.concatenate(ytr), n_classes)
#            yte = to1hot(np.concatenate(yte), n_classes)
#
#            mxtr.append(np.concatenate(xtr))
#            mxte.append(np.concatenate(xte))
#            mytr.append(ytr)
#            myte.append(yte)
#
##        np.save('logs/mxtr.npy', mxtr[0])
##        np.save('logs/mxte.npy', mxte[0])
##        np.save('logs/mytr.npy', mytr[0])
##        np.save('logs/myte.npy', myte[0])
#
#        return mxtr, mxte, mytr, myte
    def get_dataset(self, mode):
        if mode=='train':
            return self.train_data
        elif mode=='test':
            return self.test_data
        elif mode=='val':
            return self.val_data
        else: 
            print ('you should select mode in (train, test, val)')

    def data_queue(self, mode, meta_batch_size, n_classes, kshot=0, debug=False):
        data = self.get_dataset(mode)
        mxtr, mxte, mytr, myte = [], [], [], []
        # mxtr.shape : (meta_batch_size, num_classes*num_instances, 84*84*3)
        # the shape will be specified only when the meta-batch size is 1    
        for _ in range(meta_batch_size):
            tasks_ind = np.arange(len(data))
            np.random.shuffle(tasks_ind)
            class_data = data[tasks_ind[:n_classes]] # select nway classes
            tr, te, cln = [], [], []
            for n in range(n_classes):
                if kshot != 0:
                    class_len = np.amin([class_data[n].shape[0], kshot*2])
                else:
                    class_len = class_data[n].shape[0]
                cln.append(class_len)
                rnd_ind = np.arange(class_len)
                np.random.shuffle(rnd_ind)
                tr.append(class_data[n][rnd_ind[:(class_len//2)]])
                te.append(class_data[n][rnd_ind[(class_len//2):]])
            mxtr.append(np.concatenate(tr))
            mxte.append(np.concatenate(te))

#            rnd_ind = [np.random.choice(c.shape[0], c.shape[0], replace=False) for c in class_data]
#            mxtr.append(np.concatenate([cld[:rnd_ind//2] for cld in range(n_classes)]))
#            mxte.append(np.concatenate([cld[cld.shape[0]//2:] for cld in class_data]))
            #y = np.concatenate([[n]*(len(class_data[n])//2) for n in range(n_classes)])
            y = np.concatenate([[n]*(cln[n]//2) for n in range(n_classes)])
            ylen = np.shape(y)[0]
            y1hot = np.zeros([ylen, n_classes], dtype=int)
            y1hot[np.arange(ylen), y] = 1
            mytr.append(y1hot.copy())
            myte.append(y1hot.copy())

            ty = [[tasks_ind[n]]*(len(class_data[n])//2) for n in range(n_classes)]
            ty = np.concatenate(ty)

#        np.save('logs/mxtr.npy', mxtr[0])
#        np.save('logs/mxte.npy', mxte[0])
#        np.save('logs/mytr.npy', mytr[0])
#        np.save('logs/myte.npy', myte[0])
        if debug:
            return mxtr, mxte, mytr, myte, ty
        
        #only single meta-batch
        return mxtr[0], mxte[0], mytr[0], myte[0]

def to1hot(label, nway):
    data_len = np.shape(label)[0]
    yout = np.zeros([data_len, nway], dtype=int)
    yout[np.arange(data_len), label.astype(int)] = 1 
    return yout

if __name__=='__main__':
    tgen = TieredGenerator(root='./data/tiered-imagenet')
    #tgen.data_queue('train', 1, 5)
    for i in range(100):
        #tgen._data_queue('train', 1, 5)
        tgen._data_queue('train', 1, 5)
