%%% Read the segy file and extract the size of the array (the path should
%%% be where the .sgy file was downloaded).
D = read_segy_file('gob_20200731_synthetic_shot-gather.sgy');
D = D.traces;
%D = D(1:200,300:499);%Patch #1
%D = D(421:620,551:750);%Patch #2
%D = D(1421:1620,400:599);%Patch #3
D = D(1100:1299,1:200);%Patch #4
nt = size(D,1);
nr = size(D,2);
%%% Normalize the data
%D = log10(1+1e3*abs(D)).*sign(D);

%%% Visualize the original data
figure(1);
imagesc(D); colormap('redblue'); colorbar;
title('Original data'); caxis([-1000,1000]);
xlabel('Traces'); ylabel('Time samples')

%%% Seed the random number generator and the subsampling factor
rng(123456789);
p = 0.5;
%%% Create a random permutation of the receiver indices
idx = randperm(nr);
idx = sort(idx(1:round(nr*p)));
%%% Create the sampling operator (to drop the traces)
R = opKron(opRestriction(nr, idx), opDirac(nt));
%%% Take out the dropped traces
RD = R*D(:);
%%% Re-insert the traces w/ missing data
Dest_adj = reshape(R'*RD(:), nt, nr);

%%% Visualize the missing data
figure(2);
imagesc(Dest_adj); colormap('redblue'); colorbar;
title('Dropped traces'); caxis([-1000,1000]);
xlabel('Traces'); ylabel('Time samples')

%%% Create the sensing operator (we will create F, the DFT/Wavelet operator, and then
%%% take the transpose since it is an orthonormal basis
F = opDFT2(nt, nr);
%F = opWavelet2(nt,nr,'Daubechies');
%F = opWavelet2(nt,nr,'Haar');
%F2 = opCurvelet
%%% Create the caption vector 'y' using the operator RD
y = RD(:);
%%% Create the Theta operator by taking random samples of the inverse
%%% Fourier transform operator F^t
T = R*F';
%%% Set the options for the basis pursuit solver
options = spgSetParms('optTol', 5e-3, 'bpTol', 5e-3,...
    'iterations', 100, 'verbosity', 1);
%%% Now solve the \ell_1 min problem
xest = spg_bp(T, y, options);

%%% Transform the solution to the temporal domain w/ the original shape
dest = F'*xest;
Dest = reshape(dest, nt, nr);
%%% Determine the accuracy of the recover by computing the residual and the
%%% signal-to-noise ratio (SNR)
Ddiff = D - Dest;
SNR = -20*log10(norm(Ddiff(:))/norm(D(:)));

%%% Visualize the results
figure(3);
imagesc(real(Dest)); colormap('redblue'); colorbar;
title('Basis Pursuit Recovery'); caxis([-1000, 1000]);
xlabel('Traces'); ylabel('Time sample')

figure(4);
imagesc(real(Ddiff)); colormap('redblue'); colorbar;
title('Residual'); caxis([-1000, 1000]);
xlabel('Traces'); ylabel('Time sample')


