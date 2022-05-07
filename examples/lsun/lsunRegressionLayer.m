classdef lsunRegressionLayer < nnet.layer.RegressionLayer
        
    properties
        % (Optional) Layer properties.

        % Layer properties go here.
    end
 
    methods
        function layer = lsunRegressionLayer(name)           
            % (Optional) Create lsunRegressionLayer.

            % Layer constructor function goes here.
            layer.Name = name;
        end

        function loss = forwardLoss(~, Y, T)
            % Return the loss between the predictions Y and the training
            % targets T.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         loss  - Loss between Y and T

            % Layer forward loss function goes here.
            N = size(Y,4);
            loss = sum(T.^2,"all")-sum(Y.^2,"all")/N;

        end
        
    end
end
