import torch
from time import time
from utils import calc_psnr, calc_ssim, cvrt_rgb_to_y, norm, denorm, bicubic

def test(args, data_name):

    net = args['net']
    scale_factor = args['scale_factor']
    criterion = args['criterion']
    device = args['device']
    test_dataloader = args['test_dataloader']
    check_pnt_path = args['check_pnt_path']
    writer = args['writer']
    crop_out = args['crop_out']

    batch_num = len(test_dataloader)

    net.load_state_dict(torch.load(check_pnt_path))

    net.eval()

    with torch.no_grad():
        
        start = time()

        total_psnr = 0
        total_ssim = 0
        total_bicubic_psnr = 0
        total_bicubic_ssim = 0

        for iteration, (img, lbl) in enumerate(test_dataloader):
            img = norm(img.to(device))
            lbl = norm(lbl.to(device))
            out = net(img)
            loss = criterion(out, lbl)

            out_cpu = denorm(out.detach()).clamp(min=0.0, max=1.0).to('cpu')
            img_cpu = bicubic(denorm(img.detach()), scale_factor).to('cpu')
            lbl_cpu = denorm(lbl.detach()).to('cpu')
            
            writer.add_images(tag='Test Upscale/'+data_name+'/Ground Truth', img_tensor=lbl_cpu, global_step=iteration+1)
            writer.add_images(tag='Test Upscale/'+data_name+'/Model', img_tensor=out_cpu, global_step=iteration+1)
            writer.add_images(tag='Test Upscale/'+data_name+'/Bicubic', img_tensor=img_cpu, global_step=iteration+1)

            out_y_np = cvrt_rgb_to_y(out_cpu.numpy())
            img_y_np = cvrt_rgb_to_y(img_cpu.numpy())
            lbl_y_np = cvrt_rgb_to_y(lbl_cpu.numpy())

            psnr = calc_psnr(out_y_np, lbl_y_np, crop_out)
            ssim = calc_ssim(out_y_np, lbl_y_np, crop_out)
            bicubic_psnr = calc_psnr(img_y_np, lbl_y_np, crop_out)
            bicubic_ssim = calc_ssim(img_y_np, lbl_y_np, crop_out)

            total_psnr += psnr
            total_ssim += ssim
            total_bicubic_psnr += bicubic_psnr
            total_bicubic_ssim += bicubic_ssim
        

        avg_psnr = total_psnr/batch_num
        avg_ssim = total_ssim/batch_num
        avg_bicubic_psnr = total_bicubic_psnr/batch_num
        avg_bicubic_ssim = total_bicubic_ssim/batch_num

        writer.add_scalars('Test PSNR - '+data_name, {'Model PSNR': avg_psnr, 'Bicubic PSNR': avg_bicubic_psnr})
        writer.add_scalars('Test SSIM - '+data_name, {'Model SSIM': avg_ssim, 'Bicubic SSIM': avg_bicubic_ssim})
        
        end = time()

        print(f'Test PSNR: {avg_psnr:.3f} | Test SSIM: {avg_ssim:.3f} | Bicubic PSNR: {avg_bicubic_psnr:.3f} | Bicubic SSIM: {avg_bicubic_ssim:.3f} | Time: {end-start:.3f}')
