#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import time


class reactive_net(nn.Module):

    def __init__(self, use_cuda):  # , snapshot=None
        super(reactive_net, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.push_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.push_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.grasp_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.grasp_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)

        self.num_rotations = 16

        # Construct network branches for pushing and grasping
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(2048)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(64)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 3, kernel_size=1, stride=1, bias=False))
            # ('push-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(2048)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, 3, kernel_size=1, stride=1, bias=False))
            # ('grasp-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            output_prob = []
            interm_feat = []

            # Apply rotations to images
            for rotate_idx in range(self.num_rotations):
                rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                # Compute sample grid for rotation BEFORE neural network
                affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                                [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                if self.use_cuda:
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                     input_color_data.size())
                else:
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                     input_color_data.size())

                # Rotate images clockwise
                if self.use_cuda:
                    rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(), flow_grid_before,
                                                 mode='nearest')
                    rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(), flow_grid_before,
                                                 mode='nearest')
                else:
                    rotate_color = F.grid_sample(Variable(input_color_data, volatile=True), flow_grid_before,
                                                 mode='nearest')
                    rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True), flow_grid_before,
                                                 mode='nearest')

                # Compute intermediate features
                interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
                interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
                interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
                interm_feat.append([interm_push_feat, interm_grasp_feat])

                # Compute sample grid for rotation AFTER branches
                affine_mat_after = np.asarray(
                    [[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                if self.use_cuda:
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                    interm_push_feat.data.size())
                else:
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                    interm_push_feat.data.size())

                # Forward pass through branches, undo rotation on output predictions, upsample results
                output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(
                    F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest')),
                                    nn.Upsample(scale_factor=16, mode='bilinear').forward(
                                        F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after,
                                                      mode='nearest'))])

            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                 input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                 input_color_data.size())

            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before,
                                             mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before,
                                             mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before,
                                             mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before,
                                             mode='nearest')

            # Compute intermediate features
            interm_push_color_feat = self.push_color_trunk.features(rotate_color)
            interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
            interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
            interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
            interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
            interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
            self.interm_feat.append([interm_push_feat, interm_grasp_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray(
                [[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                interm_push_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                interm_push_feat.data.size())

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(
                F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest')),
                                     nn.Upsample(scale_factor=16, mode='bilinear').forward(
                                         F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after,
                                                       mode='nearest'))])

            return self.output_prob, self.interm_feat


class reinforcement_net(nn.Module):

    def __init__(self, use_cuda):  # , snapshot=None
        super(reinforcement_net, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.push_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.push_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.grasp_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.grasp_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)

        self.num_rotations = 16

        # Construct network branches for pushing and grasping
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(2048)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(64)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
            # ('push-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(2048)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
            # ('grasp-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            output_prob = []
            interm_feat = []

            # Apply rotations to images
            for rotate_idx in range(self.num_rotations):
                rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                # Compute sample grid for rotation BEFORE neural network
                affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                                [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                if self.use_cuda:
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                     input_color_data.size())
                else:
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                     input_color_data.size())

                # Rotate images clockwise
                if self.use_cuda:
                    rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(), flow_grid_before,
                                                 mode='nearest')
                    rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(), flow_grid_before,
                                                 mode='nearest')
                else:
                    rotate_color = F.grid_sample(Variable(input_color_data, volatile=True), flow_grid_before,
                                                 mode='nearest')
                    rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True), flow_grid_before,
                                                 mode='nearest')

                # Compute intermediate features
                interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
                interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
                interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
                interm_feat.append([interm_push_feat, interm_grasp_feat])

                # Compute sample grid for rotation AFTER branches
                affine_mat_after = np.asarray(
                    [[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                if self.use_cuda:
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                    interm_push_feat.data.size())
                else:
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                    interm_push_feat.data.size())

                # Forward pass through branches, undo rotation on output predictions, upsample results
                output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(
                    F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest')),
                                    nn.Upsample(scale_factor=16, mode='bilinear').forward(
                                        F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after,
                                                      mode='nearest'))])

            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray(
                [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                 input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                 input_color_data.size())

            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before,
                                             mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before,
                                             mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before,
                                             mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before,
                                             mode='nearest')

            # Compute intermediate features
            interm_push_color_feat = self.push_color_trunk.features(rotate_color)
            interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
            interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
            interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
            interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
            interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
            self.interm_feat.append([interm_push_feat, interm_grasp_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray(
                [[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                interm_push_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                interm_push_feat.data.size())

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(
                F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest')),
                                     nn.Upsample(scale_factor=16, mode='bilinear').forward(
                                         F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after,
                                                       mode='nearest'))])

            return self.output_prob, self.interm_feat


class reinforcement2d_net(nn.Module):

    def __init__(self, use_cuda):  # , snapshot=None
        super(reinforcement2d_net, self).__init__()
        self.use_cuda = use_cuda

        # Construct network branches for pushing and grasping
        # self.pushnet = nn.Sequential(OrderedDict([
        #     ('push-norm0', nn.BatchNorm2d(6)),
        #     ('push-relu0', nn.ReLU(inplace=True)),
        #     ('push-conv0', nn.Conv2d(6, 64, kernel_size=1, stride=1, bias=False)),
        #     ('push-norm1', nn.BatchNorm2d(64)),
        #     ('push-relu1', nn.ReLU(inplace=True)),
        #     ('push-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        #     # ('push-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        # ]))
        # self.graspnet = nn.Sequential(OrderedDict([
        #     ('grasp-norm0', nn.BatchNorm2d(6)),
        #     ('grasp-relu0', nn.ReLU(inplace=True)),
        #     ('grasp-conv0', nn.Conv2d(6, 64, kernel_size=1, stride=1, bias=False)),
        #     ('grasp-norm1', nn.BatchNorm2d(64)),
        #     ('grasp-relu1', nn.ReLU(inplace=True)),
        #     ('grasp-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        #     # ('grasp-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        # ]))
        #
        # # Initialize network weights
        # for m in self.named_modules():
        #     if 'push-' in m[0] or 'grasp-' in m[0]:
        #         if isinstance(m[1], nn.Conv2d):
        #             nn.init.kaiming_normal(m[1].weight.data)
        #         elif isinstance(m[1], nn.BatchNorm2d):
        #             m[1].weight.data.fill_(1)
        #             m[1].bias.data.zero_()
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-fc1', nn.Linear(6, 1024)),
            ('push-relu1', nn.ReLU()),
            ('push-fc2', nn.Linear(1024, 1)),
            ('push-relu2', nn.ReLU())
        ]))

        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-fc1', nn.Linear(6, 1024)),
            ('grasp-relu1', nn.ReLU()),
            ('grasp-fc2', nn.Linear(1024, 1)),
            ('grasp-relu2', nn.ReLU())
        ]))

        # Initialize output variable (for backprop)
        # self.interm_feat = []
        # self.output_prob = []
        self.push_predictions = []
        self.grasp_predictions = []
        self.interm_feat = []

    def forward(self, push_samples, grasp_samples, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            output_prob = []
            interm_feat = []

            # print(push_samples.shape)
            push_input = Variable(torch.from_numpy(push_samples).float())
            grasp_input = Variable(torch.from_numpy(grasp_samples).float())

            # output_prob = [self.pushnet(push_samples), self.graspnet(grasp_samples)]
            # self.pushnet = nn.Linear(6, 6)
            # self.graspnet = nn.Linear(6, 6)
            # output_prob = [self.pushnet(push_samples), self.graspnet(grasp_samples)]
            # print (output_prob)
            push_predictions, grasp_predictions = self.pushnet(push_input).detach().numpy(), self.graspnet(
                grasp_input).detach().numpy()
            # print("\npush_predictions\n", push_predictions)
            # print("\ngrasp_predictions\n", grasp_predictions)
            # exit(-1)

            return push_predictions, grasp_predictions, interm_feat

        else:
            self.push_predictions = []
            self.grasp_predictions = []
            self.interm_feat = []

            push_input = Variable(torch.from_numpy(push_samples).float())
            grasp_input = Variable(torch.from_numpy(grasp_samples).float())

            # self.output_prob = [self.pushnet(push_samples), self.graspnet(grasp_samples)]
            # exit(-1)
            self.push_predictions, self.grasp_predictions = self.pushnet(push_input).detach().numpy(), self.graspnet(
                grasp_input).detach().numpy()
            # print("\npush_predictions\n", self.push_predictions)
            # print("\ngrasp_predictions\n", self.grasp_predictions)

            return self.push_predictions, self.grasp_predictions, self.interm_feat


class reinforcement6d_net(nn.Module):

    def __init__(self, use_cuda):  # , snapshot=None
        super(reinforcement6d_net, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.push_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.push_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.grasp_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.grasp_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)

        self.num_rotations = 16

        # Construct network branches for pushing and grasping
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(2048)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(64)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
            # ('push-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))

        self.graspnet_fc = nn.Sequential(OrderedDict([
            ('grasp-fc1', nn.Linear(6, 1024)),
            ('grasp-fc1-relu1', nn.ReLU()),
            ('grasp-fc2', nn.Linear(1024, 1)),
            ('grasp-fc2-relu2', nn.ReLU())
        ]))

        # self.graspnet_conv = torchvision.models.resnet.resnet50(pretrained=True)
        self.graspnet_heat = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(2048)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.push_interm_feat = []
        self.push_output_prob = []
        self.grasp_output_prob = []
        self.grasp_output_heat = []

    def forward(self, input_color_data, input_depth_data, grasp_samples, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            grasp_output_heat = []
            grasp_output_prob = []
            push_output_prob = []
            push_interm_feat = []

            # predict grasp
            if self.use_cuda:
                grasp_output_prob = self.graspnet_fc(Variable(torch.from_numpy(grasp_samples).float().cuda())).cpu().data.numpy()
            else:
                grasp_output_prob = self.graspnet_fc(Variable(torch.from_numpy(grasp_samples).float())).detach().numpy()

            # predict push
            # Apply rotations to images
            for rotate_idx in range(self.num_rotations):
                rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                # Compute sample grid for rotation BEFORE neural network
                affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                if self.use_cuda:
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
                else:
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())

                # Rotate images clockwise
                if self.use_cuda:
                    rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(), flow_grid_before,  mode='nearest')
                    rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(), flow_grid_before, mode='nearest')
                else:
                    rotate_color = F.grid_sample(Variable(input_color_data, volatile=True), flow_grid_before, mode='nearest')
                    rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True), flow_grid_before, mode='nearest')

                # Compute intermediate features
                interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
                push_interm_feat.append([interm_push_feat])

                # Compute sample grid for rotation AFTER branches
                affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                if self.use_cuda:
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_push_feat.data.size())
                else:
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_push_feat.data.size())

                push_output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest'))])
                if rotate_idx == 0:
                    interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
                    interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                    interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
                    grasp_output_heat.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.graspnet_heat(interm_grasp_feat), flow_grid_after, mode='nearest'))])

            return push_output_prob, grasp_output_prob, grasp_output_heat, push_interm_feat

        else:
            self.push_output_prob = []
            self.grasp_output_prob = []
            self.push_interm_feat = []
            self.grasp_interm_feat = []
            self.grasp_output_heat = []

            if self.use_cuda:
                self.grasp_output_prob = self.graspnet_fc(Variable(torch.from_numpy(grasp_samples).float().cuda())).cpu().data.numpy()
            else:
                self.grasp_output_prob = self.graspnet_fc(Variable(torch.from_numpy(grasp_samples).float())).detach().numpy()

            # Apply rotations to intermediate features
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())

            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before, mode='nearest')

            # Compute intermediate features
            interm_push_color_feat = self.push_color_trunk.features(rotate_color)
            interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
            interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
            self.push_interm_feat.append([interm_push_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_push_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_push_feat.data.size())

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.push_output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest'))])

            rotate_idx = 0
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())

            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before, mode='nearest')

            interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
            interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
            interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
            self.grasp_interm_feat.append([interm_grasp_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_grasp_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_grasp_feat.data.size())

            self.grasp_output_heat.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.graspnet_heat(interm_grasp_feat), flow_grid_after, mode='nearest'))])
            # print(self.push_output_prob)
            # print(self.grasp_output_heat)

            return self.push_output_prob, self.grasp_output_prob, self.grasp_output_heat, self.push_interm_feat