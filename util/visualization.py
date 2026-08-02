import os
import numpy as np
import cv2
import sys

import torch

import matplotlib.pyplot as plt

def fast_show_mask_gpu(
        annotation,
        ax,
        random_color=False,
        bboxes=None,
        points=None,
        pointlabel=None,
        retinamask=True,
        target_height=960,
        target_width=960,
    ):
        mask_num = annotation.shape[0]
        height = annotation.shape[1]
        weight = annotation.shape[2]
        areas = torch.sum(annotation, dim=(1, 2))
        sorted_indices = torch.argsort(areas, descending=False)
        annotation = annotation[sorted_indices]
        # Find the index of the first non-zero value at each position.
        index = (annotation != 0).to(torch.long).argmax(dim=0)
        if random_color:
            color = torch.rand((mask_num, 1, 1, 3)).to(annotation.device)
        else:
            color = torch.ones((mask_num, 1, 1, 3)).to(annotation.device) * torch.tensor([
                30 / 255, 144 / 255, 255 / 255]).to(annotation.device)
        transparency = torch.ones((mask_num, 1, 1, 1)).to(annotation.device) * 0.6
        visual = torch.cat([color, transparency], dim=-1)
        mask_image = torch.unsqueeze(annotation, -1) * visual
        # Select data according to the index. The index indicates which batch's data to choose at each position, converting the mask_image into a single batch form.
        show = torch.zeros((height, weight, 4)).to(annotation.device)
        try:
            h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight), indexing='ij')
        except:
            h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight))
        indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
        # Use vectorized indexing to update the values of 'show'.
        show[h_indices, w_indices, :] = mask_image[indices]
        show_cpu = show.cpu().numpy()
        if bboxes is not None:
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b', linewidth=1))
        # draw point
        if points is not None:
            plt.scatter(
                [point[0] for i, point in enumerate(points) if pointlabel[i] == 1],
                [point[1] for i, point in enumerate(points) if pointlabel[i] == 1],
                s=20,
                c='y',
            )
            plt.scatter(
                [point[0] for i, point in enumerate(points) if pointlabel[i] == 0],
                [point[1] for i, point in enumerate(points) if pointlabel[i] == 0],
                s=20,
                c='m',
            )
        if not retinamask:
            show_cpu = cv2.resize(show_cpu, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        ax.imshow(show_cpu)

def plot_to_result(
             image,
             annotations,
             bboxes=None,
             points=None,
             point_label=None,
             mask_random_color=True,
             better_quality=True,
             retina=False,
             withContours=True) -> np.ndarray:
        if isinstance(annotations[0], dict):
            annotations = [annotation['segmentation'] for annotation in annotations]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h = image.shape[0]
        original_w = image.shape[1]
        if sys.platform == "darwin":
            plt.switch_backend("TkAgg")
        plt.figure(figsize=(original_w / 100, original_h / 100))
        # Add subplot with no margin.
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.imshow(image)
        if better_quality:
            if isinstance(annotations[0], torch.Tensor):
                annotations = np.array(annotations.cpu())
            for i, mask in enumerate(annotations):
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                annotations[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
        # if self.device == 'cpu':
        #     annotations = np.array(annotations)
        #     self.fast_show_mask(
        #         annotations,
        #         plt.gca(),
        #         random_color=mask_random_color,
        #         bboxes=bboxes,
        #         points=points,
        #         pointlabel=point_label,
        #         retinamask=retina,
        #         target_height=original_h,
        #         target_width=original_w,
        #     )
        # else:
        if isinstance(annotations[0], np.ndarray):
            annotations = torch.from_numpy(annotations)
            fast_show_mask_gpu(
                annotations,
                plt.gca(),
                random_color=mask_random_color,
                bboxes=bboxes,
                points=points,
                pointlabel=point_label,
                retinamask=retina,
                target_height=original_h,
                target_width=original_w,
            )
        if isinstance(annotations, torch.Tensor):
            annotations = annotations.cpu().numpy()
        if withContours:
            contour_all = []
            temp = np.zeros((original_h, original_w, 1))
            for i, mask in enumerate(annotations):
                if type(mask) == dict:
                    mask = mask['segmentation']
                annotation = mask.astype(np.uint8)
                if not retina:
                    annotation = cv2.resize(
                        annotation,
                        (original_w, original_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                contours, hierarchy = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour_all.append(contour)
                    M = np.mean(contour, axis=0)[0]
                    cx, cy = int(M[0]), int(M[1])

                    # text_to_draw = 'building'
                    text_to_draw = 'land use'
                    font_face = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.3
                    thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(text_to_draw, font_face, font_scale, thickness)

                    # 计算背景框的位置 (使其以文本为中心)
                    padding = 5
                    box_x1 = cx - text_width // 2 - padding
                    box_y1 = cy + baseline // 2 - text_height
                    box_x2 = box_x1 + text_width + padding * 2
                    box_y2 = box_y1 + text_height + padding * 2
                    
                    # 绘制黑色背景矩形 (带白色边框)
                    # cv2.rectangle(temp, (box_x1, box_y1), (box_x2, box_y2), (168, 100, 228), thickness=cv2.FILLED) # 填充
                    # cv2.rectangle(temp, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 1) # 边框

                    # 计算文本绘制的起始位置
                    text_x = cx - text_width // 2
                    text_y = cy + text_height // 2
                    
                    # 在矩形上绘制白色文本
                    # cv2.putText(temp, text_to_draw, (text_x, text_y), font_face, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    # cv2.putText(temp, text_to_draw, (text_x, text_y), font_face, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2)
            # cv2.drawContours(temp, contour_all, -1, (0, 0, 0), 2)
            # color = np.array([0 / 255, 0 / 255, 255 / 255, 0.8])
            color = np.array([1, 1, 1, 0.8])
            contour_mask = temp / 255 * color.reshape(1, 1, -1)
            # contour_mask = temp * color.reshape(1, 1, -1)
            plt.imshow(contour_mask)

        plt.axis('off')
        fig = plt.gcf()
        plt.draw()

        try:
            buf = fig.canvas.tostring_rgb()
        except AttributeError:
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
        cols, rows = fig.canvas.get_width_height()
        img_array = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols, 3)
        result = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        plt.close()
        return result

def plot(image,
        annotations,
        output_path,
        bboxes=None,
        points=None,
        point_label=None,
        mask_random_color=True,
        better_quality=True,
        retina=False,
        withContours=True):
        if len(annotations) == 0:
            return None
        result = plot_to_result(
               image,
               annotations, 
               bboxes, 
               points, 
               point_label, 
               mask_random_color,
               better_quality, 
               retina, 
               withContours,
        )

        path = os.path.dirname(os.path.abspath(output_path))
        if not os.path.exists(path):
                os.makedirs(path)
        result = result[:, :, ::-1]
        cv2.imwrite(output_path, result)