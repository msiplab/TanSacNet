import torch
import torch.nn as nn
import torch.autograd as autograd
from lsunLayerExceptions import InvalidDirection, InvalidTargetChannels
from lsunUtility import Direction

class LsunAtomExtension2dLayer(nn.Module):
    """
    LSUNATOMEXTENSION2DLAYER
        コンポーネント別に入力(nComponents=1のみサポート):
            nSamples x nRows x nCols x nChs
    
        コンポーネント別に出力(nComponents=1のみサポート):
            nSamples x nRows x nCols x nChs
    
    Requirements: Python 3.10/11.x, PyTorch 2.3.x
    
    Copyright (c) 2024, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
                    Faculty of Engineering, Niigata University,
                    8050 2-no-cho Ikarashi, Nishi-ku,
                    Niigata, 950-2181, JAPAN
    
    https://www.eng.niigata-u.ac.jp/~msiplab/
    """
    def __init__(self,
            name='',
            stride=[],
            direction='',
            target_channels=''):
        super(LsunAtomExtension2dLayer, self).__init__()
        self.stride = stride
        self.name = name

        # Target channels
        if target_channels in { 'Sum', 'Difference' }:
            self.target_channels = target_channels
        else:
            raise InvalidTargetChannels(
                '%s : Target should be either of Sum or Difference'\
                % self.direction
            )

        # Shift direction
        if direction in { 'Right', 'Left', 'Down', 'Up' }:
            self.direction = direction        
        else:
           raise InvalidDirection(
                '%s : Direction should be either of Right, Left, Down or Up'\
                % self.direction
            )

        # Description
        nChsTotal = self.stride[Direction.VERTICAL]*self.stride[Direction.HORIZONTAL]
        ps = nChsTotal//2+nChsTotal%2
        pa = nChsTotal//2
        self.description = direction \
            + " shift the " \
            + target_channels.lower() \
            + "-channel Coefs. " \
            + "(ps,pa) = (" \
            + str(ps) + "," \
            + str(pa) + ")"
        self.type = ''        

    def forward(self,X):
        # Number of channels
        nChsTotal = self.stride[Direction.VERTICAL]*self.stride[Direction.HORIZONTAL]
        ps = nChsTotal//2+nChsTotal%2
        pa = nChsTotal//2
        nchs = torch.tensor([ps,pa],dtype=torch.int)

        # Target channels
        if self.target_channels == 'Difference':
            target = torch.tensor((0,))
        else:
            target = torch.tensor((1,))
        # Shift direction
        if self.direction == 'Right':
            shift = torch.tensor(( 0, 0, 1, 0 ))
        elif self.direction == 'Left':
            shift = torch.tensor(( 0, 0, -1, 0 ))
        elif self.direction == 'Down':
            shift = torch.tensor(( 0, 1, 0, 0 ))
        else:
            shift = torch.tensor(( 0, -1, 0, 0 ))
        # Atom extension function
        atomext = AtomExtension2d.apply

        return atomext(X,nchs,target,shift)

class AtomExtension2d(autograd.Function):
    @staticmethod
    def forward(ctx, input, nchs, target, shift):
        ctx.mark_non_differentiable(nchs,target,shift)
        ctx.save_for_backward(nchs,target,shift)
        # Block butterfly 
        X = block_butterfly(input,nchs)
        # Block shift
        X = block_shift(X,nchs,target,shift)        
        # Block butterfly 
        return block_butterfly(X,nchs)/2.

    @staticmethod
    def backward(ctx, grad_output):
        nchs,target,shift = ctx.saved_tensors
        grad_input = grad_nchs = grad_target = grad_shift = None
        if ctx.needs_input_grad[0]:
            # Block butterfly 
            X = block_butterfly(grad_output,nchs)
            # Block shift
            X = block_shift(X,nchs,target,-shift)
            # Block butterfly 
            grad_input = block_butterfly(X,nchs)/2.
        if ctx.needs_input_grad[1]:
            grad_nchs = torch.zeros_like(nchs)
        if ctx.needs_input_grad[2]:
            grad_target = torch.zeros_like(target)
        if ctx.needs_input_grad[3]:
            grad_shift = torch.zeros_like(shift)
               
        return grad_input, grad_nchs, grad_target, grad_shift

def block_butterfly(X,nchs):
    """
    Block butterfly
    """
    ps = nchs[0]
    Xs = X[:,:,:,:ps]
    Xa = X[:,:,:,ps:]
    return torch.cat((Xs+Xa,Xs-Xa),dim=-1)

def block_shift(X,nchs,target,shift):
    """
    Block shift
    """
    ps = nchs[0]
    if target == 0: # Difference channel
        X[:,:,:,ps:] = torch.roll(X[:,:,:,ps:],shifts=tuple(shift.tolist()),dims=(0,1,2,3))
    else: # Sum channel
        X[:,:,:,:ps] = torch.roll(X[:,:,:,:ps],shifts=tuple(shift.tolist()),dims=(0,1,2,3))
    return X

"""
classdef lsunAtomExtension2dLayer < nnet.layer.Layer %#codegen
    %NSOLTATOMEXTENSION2DLAYER
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nChsTotal x nRows x nCols x nSamples
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nChsTotal x nRows x nCols x nSamples
    %
    % Requirements: MATLAB R2020a
    %
    % Copyright (c) 2020-2021, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    
    properties
        % (Optional) Layer properties.
        Stride
        Direction
        TargetChannels
        
        % Layer properties go here.
    end
    
    methods
        function layer = lsunAtomExtension2dLayer(varargin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'Stride',[])
            addParameter(p,'Direction','')
            addParameter(p,'TargetChannels','')
            parse(p,varargin{:})
            
            % Layer constructor function goes here.
            layer.Stride = p.Results.Stride;
            layer.Name = p.Results.Name;
            layer.Direction = p.Results.Direction;
            layer.TargetChannels = p.Results.TargetChannels;
            nChsTotal = prod(layer.Stride);
            layer.Description =  layer.Direction ...
                + " shift the " ...
                + lower(layer.TargetChannels) ...
                + "-channel Coefs. " ...
                + "(ps,pa) = (" ...
                + ceil(nChsTotal/2) + "," ...
                + floor(nChsTotal/2) + ")";
            
            layer.Type = '';
            
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data (n: # of components)
            % Outputs:
            %         Z           - Outputs of layer forward function
            %
            
            % Layer forward function for prediction goes here.
            dir = layer.Direction;
            %
            if strcmp(dir,'Right')
                shift = [ 0 0 1 0 ];
            elseif strcmp(dir,'Left')
                shift = [ 0 0 -1 0 ];
            elseif strcmp(dir,'Down')
                shift = [ 0 1 0 0 ];
            elseif strcmp(dir,'Up')
                shift = [ 0 -1 0 0 ];
            else
                throw(MException('NsoltLayer:InvalidDirection',...
                    '%s : Direction should be either of Right, Left, Down or Up',...
                    layer.Direction))
            end
            %
            Z = layer.atomext_(X,shift);
        end
        
        function dLdX = backward(layer, ~, ~, dLdZ, ~)
            % (Optional) Backward propagate the derivative of the loss  
            % function through the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X1, ..., Xn       - Input data
            %         Z1, ..., Zm       - Outputs of layer forward function            
            %         dLdZ1, ..., dLdZm - Gradients propagated from the next layers
            %         memory            - Memory value from forward function
            % Outputs:
            %         dLdX1, ..., dLdXn - Derivatives of the loss with respect to the
            %                             inputs
            %         dLdW1, ..., dLdWk - Derivatives of the loss with respect to each
            %                             learnable parameter
            
            % Layer forward function for prediction goes here.
            dir = layer.Direction;

            %
            if strcmp(dir,'Right')
                shift = [ 0 0 -1 0 ]; % Reverse
            elseif strcmp(dir,'Left')
                shift = [ 0 0 1 0 ];  % Reverse
            elseif strcmp(dir,'Down')
                shift = [ 0 -1 0 0 ];  % Reverse
            elseif strcmp(dir,'Up')
                shift = [ 0 1 0 0 ];  % Reverse
            else
                throw(MException('NsoltLayer:InvalidDirection',...
                    '%s : Direction should be either of Right, Left, Down or Up',...
                    layer.Direction))
            end
            %
            dLdX = layer.atomext_(dLdZ,shift);
        end
        
        function Z = atomext_(layer,X,shift)
            nChsTotal = prod(layer.Stride);
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            target = layer.TargetChannels;            
            %
            % Block butterfly
            Xs = X(1:ps,:,:,:);
            Xa = X(ps+1:ps+pa,:,:,:);
            Ys =  bsxfun(@plus,Xs,Xa);
            Ya =  bsxfun(@minus,Xs,Xa);
            % Block circular shift
            if strcmp(target,'Difference')
                Ya = circshift(Ya,shift);
            elseif strcmp(target,'Sum')
                Ys = circshift(Ys,shift);
            else
                throw(MException('NsoltLayer:InvalidTargetChannels',...
                    '%s : TaregetChannels should be either of Sum or Difference',...
                    layer.TargetChannels))
            end
            % Block butterfly
            Y =  cat(1,bsxfun(@plus,Ys,Ya),bsxfun(@minus,Ys,Ya));
            % Output
            Z = 0.5*Y; %ipermute(Y,[3 1 2 4])/2.0;
        end
        
    end

end
"""
