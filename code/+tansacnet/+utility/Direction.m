classdef Direction < matlab.System %#codegen
    %DIRECTION Constant values to indicate horizontal or vertical
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2022, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/    
    %
    properties (Constant = true)
        VERTICAL   = 1; % Vertical
        HORIZONTAL = 2; % Horizontal
        DEPTH      = 3; % Depth
    end
    
end
