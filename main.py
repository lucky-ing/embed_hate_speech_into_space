from trilpetloss import Hate_net

if __name__== '__main__':
    hate_net=Hate_net(True)
    hate_net.creat_net()
    hate_net.dataset_load()
    #hate_net.train_op_test()
    hate_net.valid_op_test()

