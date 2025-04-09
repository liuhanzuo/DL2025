import numpy as np

def testLinear(Linear_test,Linear_ans):
    x = np.array(
        [[0.41259363, -0.40173373, -0.9616683, 0.32021663, 0.30066854],[0.41259363, -0.40173373, -0.9616683, 0.32021663, 0.30066854]], dtype=np.float32)
    w = np.array([[-0.29742905, -0.4652604,  0.03716598],
                  [0.63429886,  0.46831214,  0.22899507],
                  [0.7614463,  0.45421863, -0.7652458],
                  [0.6237591,  0.71807355,  0.81113386],
                  [-0.34458044,  0.094055,  0.70938754]], dtype=np.float32)
    b = np.array([0., 0., 0.], dtype=np.float32)
    rtt = None
    try:
        lt = Linear_ans(5, 3)
        lt.params['weight'] = w
        lt.params['bias'] = b
        lt.is_init = True
        y = lt._forward(x)
        rtt = lt._backward(y)
    except BaseException as e:
        print(f'Test error: even the answer cannot pass the test, due to the error {e.__class__.__name__}: {e}')
        return True
   
    rt = None
    try:
        l = Linear_test(5, 3)
        l.params['weight'] = w
        l.params['bias'] = b
        l.is_init = True
        y = l._forward(x)
        rt = l._backward(y)
    except BaseException as e:
        print(f'The tested program failed because of {e.__class__.__name__}: {e}')
        return False

    try:
        assert l.grads['bias'].shape == lt.grads['bias'].shape
        assert (np.abs(l.grads['bias'] - lt.grads['bias']) < 1e-5).all()
    except AssertionError:
        print('bias failed')
        print('Expected:',lt.grads['bias'])
        print('Recieved:',l.grads['bias'])
        return False
    try:
        assert l.grads['weight'].shape == lt.grads['weight'].shape
        assert (np.abs(l.grads['weight'] - lt.grads['weight']) < 1e-5).all()
    except AssertionError:
        print('weight failed')
        print('Expected:',lt.grads['weight'])
        print('Recieved:',l.grads['weight'])
        return False
    try:
        assert rtt.shape == rt.shape
        assert (np.abs(rtt-rt)<1e-5).all()
    except AssertionError:
        print('return failed')
        print('Expected:',rtt)
        print('Recieved:',rt)
        return False
    print('succeed')
    return True

def TestReLU(ReLU_test,ReLU_ans):
    x = np.array([[[[-1.75957111,  0.0085911,  0.30235818],
                [-1.05931037,  0.75555462, -2.03922536],
                [0.86653209, -0.56438439, -1.68797524]],

               [[-0.74832044,  0.21611616,  0.571611],
                [-1.61335018, -0.37620906,  1.0353189],
                [-0.26074537,  1.98065489, -1.30691981]]],


              [[[0.32680334,  0.29817393,  2.25433969],
                [-0.16831957, -0.98864486,  0.36653],
                  [1.52712821,  1.19630751, -0.02024759]],

               [[0.48080474, -1.15229596, -0.95228854],
                  [-1.68168285, -2.86668484, -0.34833734],
                  [0.73179971,  1.69618114,  1.33524773]]]], dtype=np.float32)

    rett = ret = None
    try:
        lt = ReLU_ans()
        lt.is_init = True
        y = lt._forward(x)
        rett = lt._backward(x)
    except BaseException as e:
        print(f'The tested program failed because of {e.__class__.__name__}: {e}')
        return False

    try:
        l = ReLU_test()
        l.is_init = True
        y = l._forward(x)
        ret = l._backward(x)
    except BaseException as e:
        print(f'The tested program failed because of {e.__class__.__name__}: {e}')
        return False
    # print(ret)
    # print(rett)
    try:
        assert ret.shape == rett.shape
        assert(np.abs(ret-rett)<1e-5).all()
    except AssertionError:
        print('comparasion failed. Expected:\n',rett,'\n, but recieved:\n',ret)
        return False
    print('success!')

def RunFull(module,xx):
    layers = [
        module.Conv2D([2, 3, 3, 10]),
        module.ReLU(),
        module.MaxPool2D([3, 3], [2, 2]),
        module.Reshape(-1),
        module.Linear(10, 10),
        module.Tanh(),
    ]
    layers[0].params["weight"] = np.array(
        [[[[0.10034741, -0.10923018, -0.13951169,  0.02620922,
           0.11209463, -0.03927368,  0.2455522,  0.09440251,
          -0.00973911, -0.0551014],
         [-0.08009744,  0.11698803, -0.03137447, -0.22489624,
          -0.05207429,  0.12155709, -0.16216859, -0.16576429,
           0.01611499,  0.01625606],
         [0.06864581,  0.17847365,  0.11464144,  0.05670569,
           0.11361383, -0.09042443, -0.07059199, -0.13879062,
          -0.10536136,  0.06689657]],

          [[-0.208334, -0.03386239, -0.03531212, -0.00322536,
            -0.03788247, -0.09588832, -0.03761636,  0.20092505,
            0.22685647,  0.00093809],
         [0.06330945, -0.19632441, -0.09332216, -0.04350284,
           0.18709384, -0.1274559, -0.00866532,  0.24800156,
           0.0099521,  0.05766132],
         [-0.1937654,  0.16667984,  0.05263836,  0.02301866,
              -0.08030899, -0.10004608, -0.04238123,  0.09260008,
              0.19176605,  0.00532325]],

          [[0.08647767,  0.04389243,  0.06379496,  0.08922312,
            0.17485274, -0.2128848,  0.13467704, -0.1309234,
            -0.15617682,  0.03700767],
         [-0.20649141,  0.19680719,  0.06499291,  0.11411078,
              -0.2471389,  0.04823145,  0.02900326, -0.1741877,
              -0.1771956,  0.09986747],
         [-0.09256576,  0.09804958,  0.02777646, -0.05671893,
              0.21713808,  0.17450929, -0.07283475, -0.23311245,
              -0.09971547, -0.05200206]]],


         [[[0.10468353,  0.25300628, -0.07359204,  0.13409732,
           0.04759048,  0.09791245, -0.17818803, -0.05164933,
           -0.13235886, -0.07995545],
           [0.00670526,  0.08510415, -0.20128378, -0.18852185,
           -0.0909241,  0.08251962,  0.17697941,  0.0272014,
           0.18103799, -0.05881954],
           [-0.0935763, -0.07443489, -0.16799624,  0.16809557,
          -0.08239949,  0.02674822,  0.04012047,  0.01099809,
          -0.25400403,  0.24942434]],

            [[0.2070298, -0.14932613, -0.10598685,  0.01022026,
              0.20527782,  0.24701637, -0.12383634,  0.03287163,
              0.15678546, -0.05395091],
             [0.11802146, -0.17311034,  0.05143219,  0.1868667,
                0.24696055, -0.21484058, -0.03659691, -0.15090589,
                -0.02521261,  0.02439543],
             [-0.20770998, -0.10375416,  0.21839033,  0.03524392,
                -0.02175199,  0.12948939,  0.12353204, -0.23056503,
                0.10659301,  0.17326987]],

            [[-0.17062354,  0.1435208, -0.10902726, -0.09884633,
              0.08440794, -0.19848298,  0.08420925,  0.19809937,
              0.10026675, -0.03047777],
             [-0.03155725,  0.13539886,  0.03352691, -0.21201183,
                0.04222458,  0.16080765, -0.08321898,  0.21838641,
                0.1280547,  0.03782839],
             [0.12852815, -0.21495132,  0.18355937,  0.16420949,
                0.20934355, -0.18967807, -0.21360746, -0.18468067,
                -0.05139272, -0.03866057]]]], dtype=np.float32)

    layers[4].params['weight'] = np.array(
        [[0.06815682, -0.41381145, -0.32710046,  0.34138927, -0.03506786,
         0.33732942, -0.5395874,  0.056517,  0.47315797,  0.0900187],
         [-0.321956,  0.23854145, -0.13256437,  0.18445538, -0.51560444,
         0.14887139, -0.51245147,  0.26814377, -0.02967232, -0.41434735],
         [0.04670532, -0.47457483,  0.1680028,  0.54343534,  0.29511,
         0.08081549, -0.43529126,  0.21890727,  0.17655055, -0.49393934],
         [0.32019785,  0.020503, -0.08120787,  0.31569323, -0.09687106,
         -0.02078467, -0.34875813, -0.19573534,  0.37851244, -0.34297976],
         [-0.09060311,  0.53571045, -0.28854045,  0.45661694,  0.45833147,
            -0.44771242, -0.03981645,  0.00242787, -0.20411544, -0.4958647],
            [-0.2829692, -0.4430751, -0.28673285,  0.33716825,  0.43267703,
             -0.50037426, -0.21695638,  0.5264514,  0.04327536,  0.13836497],
            [-0.54164785, -0.01653088,  0.5349371, -0.13672741, -0.44142258,
             -0.04172686,  0.507196, -0.17326587,  0.32745343,  0.32736975],
            [-0.319598, -0.06203758,  0.23617937, -0.09802067, -0.3384849,
             0.51211435,  0.16513875,  0.4003412, -0.5200709, -0.2553419],
         [0.00226878, -0.47383627,  0.54009086, -0.28869098, -0.13770601,
         -0.31328425, -0.4322124, -0.29305372, -0.21842065,  0.14727412],
            [-0.23964529, -0.15086825, -0.5412125, -0.14709733,  0.03712023,
             -0.3702431,  0.10673262, -0.22659011,  0.14465407, -0.5190256]],
        dtype=np.float32)
    x=xx
    
    for l in layers:
        x = l._forward(x)

    y = x

    e = np.array(
        [[0.32257673, -0.43416652,  1.0324798, -0.19434273,  0.59407026,
          -0.19911239,  0.2908744,  0.27966267,  0.24996994, -0.97430784]],
        dtype=np.float32)
    # f = open(f'result_{module.__name__}.txt','w')
    for l in reversed(layers):
        # f.write(str(e))
        e = l._backward(e)

    dx = e

    return dx.shape ,dx

def RunConv(module,x,w,b,kernel_size = (2, 2, 2, 4),stride_size=(1,1),test_ret=False):
    l = module.Conv2D(kernel=kernel_size, padding='SAME', stride=stride_size)
    l.params['weight'] = w
    l.params['bias'] = b
    l.is_init = True
    y = l._forward(x)
    ret = l._backward(y)
    if test_ret:
        return l.grads['weight'],l.grads['bias'],ret
    else:
        return l.grads['weight'],l.grads['bias']

def RunMax(module,x,grad_in,pool_size=(2, 2),stride=(2, 2)):
    l = module.MaxPool2D(pool_size=pool_size, stride=stride)
    l.is_init = True
    y = l._forward(x)
    # print(y.shape)
    # print('---------------------------------')
    # print( np.random.rand(*y.shape))
    # print('---------------------------------')
    grad_ = l._backward(grad_in
    #    np.random.rand(*y.shape)
        # np.ones_like(y)
    )
    return grad_

def TestFull(full_test,full_ans,run=None):
    x = np.random.rand(2,2,4,4)
    anst=ans=None
    try:
        if run is None:
            anst = RunFull(full_ans,x)
        else:
            anst = run(full_ans)
    except BaseException as e:
        print(f'During test of answer, the exception {e.__class__.__name__}: {e} occurs')
        raise e
        return True
    try:
        if run is None:
            ans = RunFull(full_test,x)
        else:
            ans = run(full_test)
    except BaseException as e:
        print(f'During test of user, the exception {e.__class__.__name__}: {e} occurs')
        raise e
        return False
    # with open('result.json','w') as f:
    #     import json
    #     json.dump(anst[2].tolist(),f)
    if not isinstance(ans,tuple):
        ans = (ans,)
        anst = (anst,)
    for i,item in enumerate(anst):
        if isinstance(item,tuple):# shape
            if item != ans[i]:
                print(f'shape is wrong, Expected {item}, but recieved {ans[i]}')
                return False
        elif not((np.abs(ans[i]-item)<1e-4).all()):
            print(f'Answer no.{i}(in tuple) is wrong. Expected:\n{item}\n, but recieved:\n {ans[i]}\n\n')
            return False
    print('succeed')

def TestConv(full_test,full_ans,test_ret=False):
    # x = np.random.rand(2,2,3,3)# with batch size
    in_C = 10;kerH=3;kerW=5;out_C=10;in_H=10;in_W=10
    batch = 2
    x = np.random.rand(batch,in_C,in_H,in_W)
    # w = np.random.rand(2,2,2,4)
    w = np.random.rand(in_C,kerH,kerW,out_C)
    # b = np.random.rand(4)
    b = np.random.rand(out_C)
    
    # dic = {}
    # dic['x']=x.tolist()
    # dic['w']=w.tolist()
    # dic['b']=b.tolist()
    # # print('x',x)
    # # print('w',w)
    # # print('b',b)
    def run(f):
        return RunConv(f,x,w,b,kernel_size=(in_C,kerH,kerW,out_C),stride_size=(1,1),test_ret=test_ret)
        # return RunConv(f,x,w,b,test_ret=test_ret)
    TestFull(full_test,full_ans,run)

def TestMax(full_test,full_ans, small=False):
    edge = 10 if not small else 1
    x = np.random.rand(edge,edge,10,10)
    y = np.random.rand(edge,edge,4,4)
    def run(f):
        return RunMax(f,x,y,pool_size=(3,3),stride=(2,2))
    TestFull(full_test,full_ans,run)

if __name__ == '__main__':
    import full_jzc
    # replace "example" with YOUR NAME
    import full_lhz as full_example
    
    # Tests
    print("test linear:")
    testLinear(full_example.Linear,full_jzc.Linear)
    print("test conv:")
    TestConv(full_example, full_jzc, test_ret=True)
    TestConv(full_example, full_jzc, test_ret=False) # this is for further debugging
    print("test max pool:")
    TestMax(full_example, full_jzc)
    TestMax(full_example, full_jzc, small=True) # this is for further debugging
    print("test relu:")
    TestReLU(full_example.ReLU, full_jzc.ReLU)
    print("the final test:")
    TestFull(full_example, full_jzc)
    
    # NOTE: the final test may pass even when the sub-tests doesn't pass. You should check whether all the sub-tests pass.
    