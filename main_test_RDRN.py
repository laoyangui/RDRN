import os.path
import logging
import time
from collections import OrderedDict
import torch
from torchstat import stat
from util import utils_logger
from util import utils_image as util_image
# from util import utils_model

def main():

    utils_logger.logger_info('efficientsr_challenge', log_path='log/efficientsr_challenge.log')
    logger = logging.getLogger('efficientsr_challenge')

#    print(torch.__version__)               # pytorch version
#    print(torch.version.cuda)              # cuda version
#    print(torch.backends.cudnn.version())  # cudnn version

    # --------------------------------
    # basic settings
    # --------------------------------
    model_names = ['RDRN', 'IMDN']
    model_id = 0          # set the model name
    sf = 4   #scale
    model_name = model_names[model_id]
    logger.info('{:>16s} : {:s}'.format('Model Name', model_name))

    testsets = 'Test_Datasets'         # set path of testsets
    # testset_L = 'Set5'  # set current testing dataset; 'DIV2K_test_LR'

    # testset_L = 'Manga109_LR/x{}'.format(sf)
    # testsetout  = "Manga109"
    testset_L = 'Set5_LR/x{}'.format(sf)
    testsetout = "Set5"
    # testset_L = 'Set14_LR/x{}'.format(sf)
    # testsetout = "Set14"
    # testset_L = 'BSD100_LR/x{}'.format(sf)
    # testsetout = "BSD100"
    # testset_L = 'urban100_LR/x{}'.format(sf)
    # testsetout = "urban100"


    save_results = True
    print_modelsummary = True     # set False when calculating `Max Memery` and `Runtime`

    torch.cuda.set_device(0)      # set GPU ID
    logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------
    # define network and load model
    # --------------------------------
    if model_name == 'RDRN':
        from model.RDRN import RDRN as net
        model = net(in_nc=3, out_nc=3, nf=52, num_modules=6, upscale=sf)  # define network
        model_path = os.path.join('./checkpoints/RDRN_x4.pth')  # set model path

    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0')), strict=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()}, strict=True)

    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    # stat(model, (3, 224, 224))

    # --------------------------------
    # print model summary
    # --------------------------------
    if print_modelsummary:
        from util.utils_modelsummary import get_model_activation, get_model_flops
        input_dim = (3, 256, 256)  # set the input dimension

        activations, num_conv2d = get_model_activation(model, input_dim)
        logger.info('{:>16s} : {:<.4f} [M]'.format('#Activations', activations/10**6))
        logger.info('{:>16s} : {:<d}'.format('#Conv2d', num_conv2d))

        flops = get_model_flops(model, input_dim, False)
        logger.info('{:>16s} : {:<.4f} [G]'.format('FLOPs', flops/10**9))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        logger.info('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters/10**6))

    # --------------------------------
    # read image
    # --------------------------------
    L_path = os.path.join(testsets, testset_L)
    E_path = os.path.join(testsets, testset_L+'_'+model_name)
    util_image.mkdir(E_path)

    # record runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    logger.info('{:>16s} : {:s}'.format('Input Path', L_path))
    logger.info('{:>16s} : {:s}'.format('Output Path', E_path))
    idx = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for img in util_image.get_image_paths(L_path):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        extpng = '.png'
        logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        img_L = util_image.imread_uint(img, n_channels=3)
        img_L = util_image.uint2tensor4(img_L)
        torch.cuda.empty_cache()
        img_L = img_L.to(device)

        start.record()
        img_E = model(img_L)
        # img_E = utils_model.test_mode(model, img_L, mode=2, min_size=480, sf=sf)  # use this to avoid 'out of memory' issue.
        # logger.info('{:>16s} : {:<.3f} [M]'.format('Max Memery', torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2))  # Memery
        end.record()
        torch.cuda.synchronize()
        test_results['runtime'].append(start.elapsed_time(end))  # milliseconds


#        torch.cuda.synchronize()
#        start = time.time()
#        img_E = model(img_L)
#        torch.cuda.synchronize()
#        end = time.time()
#        test_results['runtime'].append(end-start)  # seconds

        # --------------------------------
        # (2) img_E
        # --------------------------------
        img_E = util_image.tensor2uint(img_E)

        if save_results:
            #util_image.imsave(img_E, os.path.join(E_path, img_name+ext))
            # util_image.imsave(img_E, os.path.join(E_path, img_name[:-2] + "_RDRN_x2" + extpng))

            if testsetout == "Set5":
                util_image.imsave(img_E, os.path.join(
                    "./results/SR/BI/RDRN/" + testsetout + "/x" + str(sf),
                    img_name[:-2] + "_RDRN_x" + str(sf) + extpng))
            else:
                util_image.imsave(img_E, os.path.join(
                    "./results/SR/BI/RDRN/" + testsetout + "/x" + str(sf),
                    img_name[:-8] + "_RDRN_x" + str(sf) + extpng))
            #util_image.imsave(img_E, os.path.join("/home/cgy/Desktop/RCAN-master/RCAN_TestCode/SR/BI/RDRN/" + testsetout + "/x2", img_name[:-8] + "_RDRN_x2" + extpng))

        ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_path, ave_runtime))


if __name__ == '__main__':

    main()
