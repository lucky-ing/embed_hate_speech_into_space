import tensorflow as tf
import numpy as np
import pandas as pd
import re
class Hate_net(object):
    def __init__(self,training=True):
        self.input_dimension=50
        self.input_lens=30
        self.hidden_size=128
        self.num_layer=5
        self.cell_size=16
        self.triplet_loss_margin=1.0
        self.training=training
        self.batch_size = 144
        '''if training:
            self.batch_size=144
        else:
            self.batch_size=1'''
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        self.cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size,initializer=tf.truncated_normal_initializer)
        self.cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size,initializer=tf.truncated_normal_initializer)
        self.input_embedding=tf.placeholder(tf.float32,shape=[self.batch_size,self.input_lens,self.input_dimension])

        self.output_label=tf.placeholder(tf.int32,[self.batch_size])


    def dataset_load(self):
        print("load datasets ...")
        if self.training:
            df=pd.read_csv('labeled_data.csv')
        else:
            df = pd.read_csv('datasets/valid_data.csv')
        self.tweets = df['tweet'].values
        self.speech_class = df['class'].values
        self.hate_ids = []
        self.offensive_ids = []
        self.neither_ids = []
        print('pre-processing...')
        emoji = re.compile("&#[0-9]*")
        amp = re.compile("&amp")
        name = re.compile("@[a-z,A-Z,0-9,_]*")
        http_link = re.compile("http://t.co[a-z,A-Z,/,0-9]*")
        token = re.compile("[,.;<>:]")
        for i in range(len(self.speech_class)):
            if self.speech_class[i] == 0:
                self.hate_ids.append(i)
            if self.speech_class[i] == 1:
                self.offensive_ids.append(i)
            if self.speech_class[i] == 2:
                self.neither_ids.append(i)
        for i in range(len(self.tweets)):
            tweet_temp=self.tweets[i]
            tweet_temp = re.sub(emoji, '', tweet_temp).lower().strip()
            tweet_temp = re.sub(name, '', tweet_temp)
            tweet_temp = re.sub(http_link, '', tweet_temp)
            tweet_temp = re.sub(amp, '', tweet_temp)
            tweet_temp = re.sub(token, '', tweet_temp)
            self.tweets[i]=tweet_temp
        self.tweets_data=[]
        self.hate_sample_size=len(self.hate_ids)*6
        self.offensive_sample_size=len(self.offensive_ids)
        self.neither_sample_size=len(self.neither_ids)*2
        print('hate sample size:',len(self.hate_ids),'after extend size:',self.hate_sample_size)
        print('offensive sample size',len(self.offensive_ids),'after extend size',self.offensive_sample_size)
        print('neither sample size',len(self.neither_ids),'after extend size',self.neither_sample_size)
        def load_vectors_glove(file_name):
            print("load vectors...")
            with open(file_name, 'r') as f:
                words = set()
                word_to_vec_map = {}

                for line in f:
                    line = line.strip().split()
                    curr_word = line[0]
                    words.add(curr_word)
                    word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            print("done ...")
            return words, word_to_vec_map


        words,word_to_vec_map=load_vectors_glove('glove.6B.50d.txt')
        for i in range(len(self.tweets)):
            tweet_temp = self.tweets[i]
            vector_temp=[]
            zero_vec=np.zeros([self.input_dimension])
            for i_charac in tweet_temp.split():
                if i_charac in words:
                    vector_temp.append(list(word_to_vec_map[i_charac]))
                if vector_temp.__len__()>=self.input_lens:
                    #print("over length!")
                    break
            for i in range(vector_temp.__len__(),self.input_lens):
                vector_temp.append(list(zero_vec))
            self.tweets_data.append(vector_temp)

        print("done ...")
    def creat_net(self):
        print("creating network...")
        x_lenth=[self.input_lens]*self.batch_size
        self.bid_output,bid_state=tf.nn.bidirectional_dynamic_rnn(self.cell_fw,self.cell_bw,self.input_embedding,
                                                                  sequence_length=x_lenth,dtype=tf.float32)
        
        lstm_outputs = tf.concat([self.bid_output[0], self.bid_output[1]], -1)
        self.bid_shape=tf.shape(lstm_outputs)
        lstm_outputs_=tf.layers.flatten(lstm_outputs)
        dense0=tf.layers.dense(lstm_outputs_,512,activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.002))
        dropout=tf.layers.dropout(dense0,rate=0.3,training=(False))
        self.output=tf.layers.dense(dropout,256,activation=tf.nn.l2_normalize,kernel_initializer=tf.truncated_normal_initializer,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.002))

        '''anc, pos, neg = self.output[::3, :], self.output[1::3, :], self.output[2::3, :]
        # 欧式距离
        pos_dist = tf.reduce_sum(tf.square(anc - pos), axis=-1, keepdims=True)
        neg_dist = tf.reduce_sum(tf.square(anc - neg), axis=-1, keepdims=True)
        basic_loss = pos_dist - neg_dist + self.triplet_loss_margin
        self.triplet_loss = tf.maximum(basic_loss, 0.0)'''
        self.triplet_loss=tf.contrib.losses.metric_learning.triplet_semihard_loss(self.output_label,self.output,self.triplet_loss_margin)
        self.train_op=tf.train.AdamOptimizer(0.01).minimize(self.triplet_loss)
        print("done ...")
    def generate_dataset(self,batch_size=18):
        import random
        return_x_datas = []
        return_y_datas = []
        first_class=0
        second_class=0
        select_class=0
        for i in range(batch_size):
            '''i_temp=i%3
            if i_temp==0:
                first_class=random.sample([0,1,2],1)[0]
                select_class=first_class
            if i_temp==1:
                select_class=first_class
            if i_temp==2:
                second_class=random.sample([0,1,2],1)[0]
                while second_class==first_class:
                    second_class = random.sample([0, 1, 2],1)[0]
                select_class=second_class
            tweet_samp_id=0
            if select_class==0:
                tweet_samp_id=random.sample(self.hate_ids,1)[0]
            if select_class==1:
                tweet_samp_id = random.sample(self.offensive_ids,1)[0]
            if select_class==2:
                tweet_samp_id= random.sample(self.neither_ids,1)[0]'''
            class_index=random.random()*(self.hate_sample_size+self.offensive_sample_size+self.neither_sample_size)
            if class_index <self.hate_sample_size:
                select_class=0
                tweet_samp_id=random.sample(self.hate_ids,1)[0]
            if class_index>=self.hate_sample_size and class_index<(self.hate_sample_size+self.offensive_sample_size):
                select_class=1
                tweet_samp_id=random.sample(self.offensive_ids,1)[0]
            if class_index>=(self.hate_sample_size+self.offensive_sample_size):
                select_class=2
                tweet_samp_id=random.sample(self.neither_ids,1)[0]
            
            return_x_datas.append(self.tweets_data[tweet_samp_id])
            return_y_datas.append(select_class)
        return np.array(return_x_datas), np.array(return_y_datas)
    def get_valid_f1_score(self,sess):
        print("load datasets ...")
        df = pd.read_csv('datasets/valid_data.csv')
        tweets = df['tweet'].values
        speech_class = df['class'].values
        hate_ids = []
        offensive_ids = []
        neither_ids = []
        emoji = re.compile("&#[0-9]*")
        amp = re.compile("&amp")
        name = re.compile("@[a-z,A-Z,0-9,_]*")
        http_link = re.compile("http://t.co[a-z,A-Z,/,0-9]*")
        token = re.compile("[,.;<>:]")
        for i in range(len(speech_class)):
            if speech_class[i] == 0:
                hate_ids.append(i)
            if speech_class[i] == 1:
                offensive_ids.append(i)
            if speech_class[i] == 2:
                neither_ids.append(i)
        for i in range(len(tweets)):
            tweet_temp = tweets[i]
            tweet_temp = re.sub(emoji, '', tweet_temp).lower().strip()
            tweet_temp = re.sub(name, '', tweet_temp)
            tweet_temp = re.sub(http_link, '', tweet_temp)
            tweet_temp = re.sub(amp, '', tweet_temp)
            tweet_temp = re.sub(token, '', tweet_temp)
            tweets[i] = tweet_temp
        tweets_data = []
        hate_sample_size = len(hate_ids) * 2
        offensive_sample_size = len(offensive_ids)
        neither_sample_size = len(neither_ids)

        def load_vectors_glove(file_name):
            print("load vectors...")
            with open(file_name, 'r') as f:
                words = set()
                word_to_vec_map = {}

                for line in f:
                    line = line.strip().split()
                    curr_word = line[0]
                    words.add(curr_word)
                    word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            print("done ...")
            return words, word_to_vec_map

        words, word_to_vec_map = load_vectors_glove('glove.6B.50d.txt')
        for i in range(len(tweets)):
            tweet_temp = tweets[i]
            vector_temp = []
            zero_vec = np.zeros([self.input_dimension])
            for i_charac in tweet_temp.split():
                if i_charac in words:
                    vector_temp.append(list(word_to_vec_map[i_charac]))
                if vector_temp.__len__() >= self.input_lens:
                    print("over length!")
                    break
            for i in range(vector_temp.__len__(), self.input_lens):
                vector_temp.append(list(zero_vec))
            tweets_data.append(vector_temp)
        import tqdm
        label_hate_vec = np.zeros([256])
        total_sample_vec = []
        total_sample_label = []
        sample_batch = []
        sample_lables_batch = []
        for sample_id in tqdm.tqdm(range(len(tweets_data))):
            sample_batch.append(tweets_data[sample_id])
            sample_lables_batch.append(int(speech_class[sample_id]))
            if sample_batch.__len__() == self.batch_size:
                output_data_vec = sess.run(self.output,
                                           feed_dict={self.input_embedding: np.array(sample_batch)})
                total_sample_vec.extend(output_data_vec)
                total_sample_label.extend(sample_lables_batch)
                sample_batch.clear()
                sample_lables_batch.clear()
        if sample_batch.__len__() != 0:
            for i in range(sample_batch.__len__(), self.batch_size):
                sample_batch.append(np.zeros([self.input_lens, self.input_dimension]))
            output_data_vec = sess.run(self.output,
                                       feed_dict={self.input_embedding: np.array(sample_batch)})
            total_sample_vec.extend(output_data_vec[:sample_lables_batch.__len__()])
            total_sample_label.extend(sample_lables_batch)
            sample_batch.clear()
            sample_lables_batch.clear()
        label_offensive_vec = np.zeros([256])
        label_neither_vec = np.zeros([256])
        for vec_id in range(total_sample_vec.__len__()):
            if total_sample_label[vec_id] == 0:
                label_hate_vec += total_sample_vec[vec_id]
            if total_sample_label[vec_id] == 1:
                label_offensive_vec += total_sample_vec[vec_id]
            if total_sample_label[vec_id] == 2:
                label_neither_vec += total_sample_vec[vec_id]
        label_hate_vec /= len(hate_ids)
        label_offensive_vec /= len(offensive_ids)
        label_neither_vec /= len(neither_ids)
        label_data = [label_hate_vec, label_offensive_vec, label_neither_vec]
        prediction_result = []
        for vec_id in range(total_sample_vec.__len__()):
            prediction_result.append(self.get_min_distance_class(total_sample_vec[vec_id], label_data))
        import sklearn
        f1_score = sklearn.metrics.f1_score(total_sample_label, prediction_result, average='macro')
        print(sklearn.metrics.classification_report(total_sample_label, prediction_result, [0, 1, 2],
                                                    ['hate speech', 'offensive speech', 'neither']))
        return f1_score

    def plot_with_label3d(self,hight_dim_embs, labels, filename='tsne3d.png'):
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from sklearn.manifold import TSNE
        plt.rcParams['font.family'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        print(matplotlib.get_cachedir())
        tsne = TSNE(perplexity=30, n_components=3, init='pca', n_iter=5000)
        # plot_only = len(labels)
        print(len(hight_dim_embs))
        low_dim_embs = tsne.fit_transform(hight_dim_embs)
        # assert low_dim_embs>=len(labels) ,'more labels than ambeddings'
        #fig=plt.figure(figsize=(18,18))
        fig = plt.figure()
        #ax = plt.subplot(111, projection='3d')
        ax=Axes3D(fig)
        for i, label in enumerate(labels):
            x, y, z = low_dim_embs[i, :]
            print(label)
            if label == 0:
                plt.scatter(x, y,z, c='r')
            if label==1:
                plt.scatter(x, y,z, c='b')
            if label==2:
                plt.scatter(x, y,z, c='y')
            #ax.text(x, y, z, label)
            # ax.annotate(label,xy=(x,y),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')
        plt.savefig(filename)
        plt.show()
    def plot_with_label2d(self,hight_dim_embs, labels, filename='tsne2d.png'):
        import matplotlib
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        plt.rcParams['font.family'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        print(matplotlib.get_cachedir())
        tsne = TSNE(perplexity=50, n_components=3, init='pca', n_iter=20000)
        # plot_only = len(labels)
        print(len(hight_dim_embs))
        low_dim_embs = tsne.fit_transform(hight_dim_embs)
        # assert low_dim_embs>=len(labels) ,'more labels than ambeddings'
        plt.figure(figsize=(18, 18))
        # ax=plt.subplot(111,projection='3d')
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            #print(label)
            if label == 0:
                plt.scatter(x, y, c='r')
            if label==1:
                plt.scatter(x, y, c='b')
            if label==2:
                plt.scatter(x, y, c='y')
            # ax.text(x,y,z,label)
            #plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.savefig(filename)
        plt.show()
    def get_min_distance_class(self,input_data,label_vector):

        min_distance=1000000
        min_distance_index=-1
        for vec_index in range(len(label_vector)):
            distance=np.sum(np.square(label_vector[vec_index]-input_data))
            if distance<min_distance:
                min_distance=distance
                min_distance_index=vec_index
        return min_distance_index

    def valid_op_test(self):
        import tqdm
        with tf.Session() as sess:
            #print(sess.run(self.bid_shape))
            init_ = tf.global_variables_initializer()
            sess.run(init_)
            saver = tf.train.Saver()
            print('loading ckpt ...')
            checkpoint = tf.train.get_checkpoint_state("model")
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            print("starting prediction hate label")
            total_hate_vec=[]
            label_hate_vec=np.zeros([256])
            total_sample_vec=[]
            total_sample_label=[]
            sample_batch=[]
            sample_lables_batch=[]
            for sample_id in tqdm.tqdm(range(len(self.tweets_data))):
                sample_batch.append(self.tweets_data[sample_id])
                sample_lables_batch.append(int(self.speech_class[sample_id]))
                if sample_batch.__len__()==self.batch_size:
                    output_data_vec = sess.run(self.output,
                                               feed_dict={self.input_embedding: np.array(sample_batch)})
                    total_sample_vec.extend(output_data_vec)
                    total_sample_label.extend(sample_lables_batch)
                    sample_batch.clear()
                    sample_lables_batch.clear()
            if sample_batch.__len__()!=0:
                for i in range(sample_batch.__len__(),self.batch_size):
                    sample_batch.append(np.zeros([self.input_lens,self.input_dimension]))
                output_data_vec = sess.run(self.output,
                                           feed_dict={self.input_embedding: np.array(sample_batch)})
                total_sample_vec.extend(output_data_vec[:sample_lables_batch.__len__()])
                total_sample_label.extend(sample_lables_batch)
                sample_batch.clear()
                sample_lables_batch.clear()
            label_offensive_vec = np.zeros([256])
            label_neither_vec = np.zeros([256])
            hate_num=0
            offensive_num=0
            neith_num=0
            tsne_data=[]
            tsne_label=[]
            max_num=500
            for vec_id in range(total_sample_vec.__len__()):
                if total_sample_label[vec_id]==0:
                    if hate_num<max_num:
                        tsne_data.append(total_sample_vec[vec_id])
                        tsne_label.append(total_sample_label[vec_id])
                        hate_num+=1
                    label_hate_vec += total_sample_vec[vec_id]
                if total_sample_label[vec_id]==1:
                    if offensive_num<max_num:
                        tsne_data.append(total_sample_vec[vec_id])
                        tsne_label.append(total_sample_label[vec_id])
                        offensive_num+=1
                    label_offensive_vec += total_sample_vec[vec_id]
                if total_sample_label[vec_id]==2:
                    if neith_num<max_num:
                        tsne_data.append(total_sample_vec[vec_id])
                        tsne_label.append(total_sample_label[vec_id])
                        neith_num+=1
                    label_neither_vec += total_sample_vec[vec_id]
            label_hate_vec /= len(self.hate_ids)
            label_offensive_vec /= len(self.offensive_ids)
            label_neither_vec /= len(self.neither_ids)
            #label_data = [label_hate_vec, label_offensive_vec, label_neither_vec]
            #label_array=np.array(label_data)
            #np.save('label.npy',label_array)

            label_data=np.load('label.npy')
            #f=open('deep.txt','w')
            #self.plot_with_label3d(tsne_data,tsne_label)
            # using kmean cluster
            from sklearn.cluster import KMeans
            #k_mean_ft = KMeans(n_clusters=3,random_state=2)
            #k_mean_ft.fit_predict(np.array(total_sample_vec))
            #label_data=k_mean_ft.cluster_centers_
            #label_data=[label_data[0],label_data[1],label_data[2]]
            prediction_result=[]
            for vec_id in range(total_sample_vec.__len__()):
                #print(total_sample_vec[vec_id])
                #a_vec_distance=np.sqrt(np.sum(np.square(total_sample_vec[vec_id]-label_data[0])))
                #b_vec_distance = np.sqrt(np.sum(np.square(total_sample_vec[vec_id] - label_data[2])))
                #c_vec_distance = np.sqrt(np.sum(np.square(total_sample_vec[vec_id] - label_data[1])))
                #f.write(str(total_sample_label[vec_id])+' '+str(a_vec_distance)+' '+str(b_vec_distance)+' '+str(c_vec_distance)+'\n')
                prediction_result.append(self.get_min_distance_class(total_sample_vec[vec_id],label_data))
            '''for hata_id in tqdm.tqdm(self.hate_ids):
                input_data_x=np.array([self.tweets_data[hata_id]])
                output_data_vec = sess.run(self.output,
                             feed_dict={self.input_embedding: input_data_x})
                total_hate_vec.append(output_data_vec[0])
                label_hate_vec+=output_data_vec[0]
            label_hate_vec/= len(self.hate_ids)
            print("starting prediction offensive label")
            total_offensive_vec = []
            label_offensive_vec = np.zeros([256])
            for offensive_id in tqdm.tqdm(self.offensive_ids):
                input_data_x = np.array([self.tweets_data[offensive_id]])
                output_data_vec = sess.run(self.output,
                                           feed_dict={self.input_embedding: input_data_x})
                total_offensive_vec.append(output_data_vec[0])
                label_offensive_vec += output_data_vec[0]
            label_offensive_vec /= len(self.offensive_ids)
            print("starting prediction neither label")
            total_neither_vec = []
            label_neither_vec = np.zeros([256])
            for neither_id in tqdm.tqdm(self.neither_ids):
                input_data_x = np.array([self.tweets_data[neither_id]])
                output_data_vec = sess.run(self.output,
                                           feed_dict={self.input_embedding: input_data_x})
                total_neither_vec.append(output_data_vec[0])
                label_neither_vec += output_data_vec[0]
            label_neither_vec /= len(self.neither_ids)
            total_num=len(total_hate_vec)+len(total_offensive_vec)+len(total_neither_vec)

            label_data=[label_hate_vec,label_offensive_vec,label_neither_vec]
            print("label data ",label_data)
            result_prediction=[]
            result_label=[]
            true_hate_num = 0
            for hate_vec in total_hate_vec:
                if self.get_min_distance_class(hate_vec,label_data)==0:
                    true_hate_num+=1
                result_prediction.append(self.get_min_distance_class(hate_vec,label_data))
                result_label.append(0)
            #print(" hate precision : ",true_hate_num/len(total_hate_vec))
            true_offensive_num = 0
            for offensive_vec in total_offensive_vec:
                if self.get_min_distance_class(offensive_vec,label_data)==1:
                    true_offensive_num+=1
                result_prediction.append(self.get_min_distance_class(offensive_vec, label_data))
                result_label.append(1)
            #print(" offensive precision : ", true_offensive_num / len(total_offensive_vec))
            true_neither_num = 0
            for neither_vec in total_neither_vec:
                if self.get_min_distance_class(neither_vec,label_data)==2:
                    true_neither_num+=1
                result_prediction.append(self.get_min_distance_class(neither_vec, label_data))
                result_label.append(2)
            #print(" neither precision : ", true_neither_num / len(total_neither_vec))
            #print(" over all precision : ",(true_neither_num+true_offensive_num+true_hate_num)/total_num)'''
            #f.close()
            import sklearn
            #print(total_sample_label)
            #print(prediction_result)
            f1_score=sklearn.metrics.f1_score(total_sample_label,prediction_result,average='macro')
            print(sklearn.metrics.classification_report(total_sample_label,prediction_result,[0,1,2],['hate speech','offensive speech','neither']))
            print('f1-score',f1_score)
    def train_op_test(self):
        with tf.Session() as sess:
            init_=tf.global_variables_initializer()
            sess.run(init_)
            saver=tf.train.Saver()
            print('loading ckpt ...')
            checkpoint = tf.train.get_checkpoint_state("model")
            saver.restore(sess,checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            saver.restore(sess,checkpoint.model_checkpoint_path)
            f1_score_last=0.0
            for i in range(1000000):
                input_data_x,input_data_y=self.generate_dataset(self.batch_size)
                #print(np.shape(input_data_x) )
                _ = sess.run(self.train_op,feed_dict={self.input_embedding:input_data_x,self.output_label:input_data_y})
                if i % 500==0:
                    output_=sess.run(self.output,feed_dict={self.input_embedding:input_data_x,self.output_label:input_data_y})
                    #print(sess.run(self.bid_shape))
                    #print(np.shape(output_))
                    #print( output_)
                    #print(input_data_y)
                    #print(np.shape(output_))
                    #print(output_)
                    pos_=np.sum(np.square(output_[0]-output_[1]))
                    neg_=np.sum(np.square(output_[0]-output_[2]))
                    print(pos_,neg_)
                    print("step ",i,' loss ',sess.run([self.triplet_loss],feed_dict={self.input_embedding:input_data_x,self.output_label:input_data_y}))
                if i%10000==0:
                    f1_score=self.get_valid_f1_score(sess)
                    if f1_score>f1_score_last:
                        f1_score_last=f1_score
                        print('./model/%d_save_model_%.04f.ckpt'%(i,f1_score))
                        saver.save(sess,'./model/%d_save_model_%.04f.ckpt'%(i,f1_score))
                    else:
                        checkpoint = tf.train.get_checkpoint_state("model")
                        saver.restore(sess,checkpoint.model_checkpoint_path)
                        print("Successfully loaded:", checkpoint.model_checkpoint_path)
                        saver.restore(sess,checkpoint.model_checkpoint_path)




