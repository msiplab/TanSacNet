classdef tripledecompTestCase < matlab.unittest.TestCase
    %TRIPLEDECOMPTESTCASE Tests for the triple decomposition x LSUN experiments
    %
    % Run
    %   cd code/examples/tripledecomp
    %   matlab -licmode onlinelicensing -batch ...
    %       "setup; results = runtests('tripledecompTestCase.m'); assertSuccess(results)"
    %
    % The first group of tests checks the three invariants of the repository
    % convention (unitarity, Parseval, perfect reconstruction) on the LSUN
    % instance that these experiments construct.  The remaining tests cover
    % the helpers introduced here.  No test needs the cylinder data.

    properties (Constant)
        Stride = [2 2]
        OverlappingFactor = [3 3]
        InputSize = [8 8]
    end

    methods (TestClassSetup)
        function addTansacnetPath(testCase)
            if isempty(which('tansacnet.lsun.fcn_createlsunlgraph2d'))
                thisdir = fileparts(mfilename('fullpath'));
                ccd = pwd;
                cd(fullfile(thisdir,'..','..'))
                setpath
                cd(ccd)
            end
            testCase.addTeardown(@() rng('default'));
        end
    end

    methods (Access = private)
        function [ana,syn] = buildPair(testCase)
            import tansacnet.lsun.*
            analgraph = fcn_createlsunlgraph2d([], ...
                'InputSize',testCase.InputSize, ...
                'Stride',testCase.Stride, ...
                'OverlappingFactor',testCase.OverlappingFactor, ...
                'NumberOfVanishingMoments',true, ...
                'Mode','Analyzer');
            synlgraph = fcn_createlsunlgraph2d([], ...
                'InputSize',testCase.InputSize, ...
                'Stride',testCase.Stride, ...
                'OverlappingFactor',testCase.OverlappingFactor, ...
                'NumberOfVanishingMoments',true, ...
                'Mode','Synthesizer');
            ana = dlupdate(@double,dlnetwork(analgraph));
            % Randomise the angles so that the invariants are not tested at
            % the trivial DCT initialisation only.
            rng(0)
            for i = 1:height(ana.Learnables)
                if ana.Learnables.Parameter(i) == "Angles"
                    ana.Learnables.Value(i) = cellfun( ...
                        @(x) x + 0.1*randn(size(x)), ...
                        ana.Learnables.Value(i),'UniformOutput',false);
                end
            end
            synlgraph = fcn_cpparamsana2syn(synlgraph,layerGraph(ana));
            syn = dlupdate(@double,dlnetwork(synlgraph));
        end

        function C = analyze(~,ana,x)
            dlx = dlarray(x,'SSCB');
            [o1,o2] = predict(ana,dlx);
            if size(o1,3) == 1
                C = cat(3,o1,o2);
            else
                C = cat(3,o2,o1);
            end
            C = double(extractdata(C));
        end
    end

    methods (Test)

        function testUnitarity(testCase)
            % Invariant 1: the analysis operator is unitary, D*D' = I.
            [ana,~] = testCase.buildPair();
            n = prod(testCase.InputSize);
            D = zeros(n,n);
            for i = 1:n
                e = zeros(testCase.InputSize);
                e(i) = 1;
                C = testCase.analyze(ana,reshape(e,[testCase.InputSize 1 1]));
                D(:,i) = C(:);
            end
            err = norm(D*D.' - eye(n),'fro');
            testCase.verifyLessThan(err,1e-10, ...
                sprintf('||D*D''-I||_F = %g',err))
        end

        function testParseval(testCase)
            % Invariant 2: the transform preserves energy.
            [ana,~] = testCase.buildPair();
            rng(1)
            y = randn([testCase.InputSize 1 4]);
            C = testCase.analyze(ana,y);
            ey = sum(y.^2,'all');
            ec = sum(C.^2,'all');
            testCase.verifyLessThan(abs(ey-ec),1e-8*ey, ...
                sprintf('energy mismatch %g relative to %g',abs(ey-ec),ey))
        end

        function testPerfectReconstruction(testCase)
            % Invariant 3: with every channel kept the synthesis inverts the
            % analysis.
            [ana,syn] = testCase.buildPair();
            rng(2)
            y = randn([testCase.InputSize 1 3]);
            dly = dlarray(y,'SSCB');
            [o1,o2] = predict(ana,dly);
            yhat = double(extractdata(predict(syn,o1,o2)));
            err = norm(yhat(:)-y(:));
            testCase.verifyLessThan(err,1e-8*norm(y(:)), ...
                sprintf('reconstruction error %g',err))
        end

        function testPhaseAvgRecoversKnownField(testCase)
            % A travelling wave with a known phase must be recovered by the
            % phase average up to the discretisation of the bins.
            nBins = 16;
            nPeriods = 12;
            nT = nBins*nPeriods;
            [xx,yy] = meshgrid(linspace(0,2*pi,24),linspace(0,2*pi,16));
            t = (0:nT-1)/nBins*2*pi;
            base = sin(xx) + 0.3*cos(2*yy);
            wave = @(ph) base + cos(xx - ph) .* (1+0.2*sin(yy));
            Y = zeros(numel(xx),nT);
            for k = 1:nT
                Y(:,k) = reshape(wave(t(k)),[],1);
            end
            rng(3)
            Yn = Y + 0.01*randn(size(Y));

            [muPhase,phaseIdx] = fcn_phaseavg(Yn,nBins);

            % Every bin must be populated, and roughly evenly: the phase
            % advances at a constant rate, so no bin may collect twice the
            % nominal number of snapshots or half of it.
            counts = accumarray(phaseIdx(:),1,[nBins 1]);
            testCase.verifyEqual(numel(counts),nBins)
            testCase.verifyGreaterThan(min(counts),nPeriods/2)
            testCase.verifyLessThan(max(counts),2*nPeriods)

            % Each bin average must match the clean field of that bin, up to
            % the averaged noise.
            errs = zeros(nBins,1);
            for k = 1:nBins
                truth = mean(Y(:,phaseIdx==k),2);
                errs(k) = norm(muPhase(:,k)-truth)/norm(truth);
            end
            testCase.verifyLessThan(max(errs),5e-2, ...
                sprintf('worst bin error %g',max(errs)))
        end

        function testDivFreeProjIsDivergenceFree(testCase)
            rng(4)
            u = randn(32,48);
            v = randn(32,48);
            [uP,vP] = fcn_divfreeproj(u,v);

            relDiv = norm(fcn_specdiv(uP,vP),'fro')/norm([uP; vP],'fro');
            testCase.verifyLessThan(relDiv,1e-10, ...
                sprintf('relative divergence %g',relDiv))

            % The projection must also remove the divergence that was there
            testCase.verifyGreaterThan( ...
                norm(fcn_specdiv(u,v),'fro')/norm([u; v],'fro'),1e-3)
        end

        function testDivFreeProjIsIdempotent(testCase)
            rng(5)
            u = randn(16,16);
            v = randn(16,16);
            [uP,vP] = fcn_divfreeproj(u,v);
            [uPP,vPP] = fcn_divfreeproj(uP,vP);
            err = norm([uPP-uP; vPP-vP],'fro')/norm([uP; vP],'fro');
            testCase.verifyLessThan(err,1e-12, ...
                sprintf('projection is not idempotent, error %g',err))
        end

        function testResidualIsOrthogonalToBasePointField(testCase)
            % The residual of the phase-averaged base-point field must have
            % zero mean within every phase bin, which is what makes it the
            % fluctuation of the triple decomposition.
            nBins = 8;
            nPeriods = 20;
            nT = nBins*nPeriods;
            [xx,~] = meshgrid(linspace(0,2*pi,20),linspace(0,2*pi,12));
            t = (0:nT-1)/nBins*2*pi;
            Y = zeros(numel(xx),nT);
            for k = 1:nT
                Y(:,k) = reshape(cos(xx - t(k)),[],1);
            end
            rng(6)
            Y = Y + 0.1*randn(size(Y));

            [muPhase,phaseIdx] = fcn_phaseavg(Y,nBins);
            R = Y - muPhase(:,phaseIdx);
            worst = 0;
            for k = 1:nBins
                worst = max(worst,norm(mean(R(:,phaseIdx==k),2)));
            end
            testCase.verifyLessThan(worst,1e-10, ...
                sprintf('per-bin residual mean %g',worst))
        end

        function testEnergyConcIsUnityWithoutTruncation(testCase)
            % With every channel retained the energy concentration is one.
            rng(7)
            C = randn(4,5,16,3);
            [ec,~,dcNorm] = fcn_energyconc(C,ones(16,1));
            testCase.verifyEqual(ec,1,'AbsTol',1e-12)
            testCase.verifyGreaterThan(dcNorm,0)
            testCase.verifyLessThan(dcNorm,1)
        end

    end
end
