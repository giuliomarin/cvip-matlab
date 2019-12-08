close all; clear
disp('----------------------------')

%% tempo continuo

A = input('Ampiezza seno: ');
f0 = input('Frequenza [Hz]: ');
tp = 1/f0;    % periodo
fprintf('Periodo: %g s\n',tp)
ntp = input('Numero di campioni per periodo: ');
Tc = tp/ntp;  % tempo di campionamento
Fc = 1/Tc;
fprintf('Frequenza di campionamento: %g Hz\n',Fc)
np = input('Numero di periodi: ');
n = 0:np*ntp-1;
t = n*Tc;
y = A*sin(2*pi*f0*t);
figure
plot(t,y)
title('Segnale a tempo continuo')

% trasformata di Fourier

N = length(y);
Y = 1/N*abs(fftshift(fft(y)));
f = -Fc/2:Fc/N:Fc/2-Fc/N;
figure
plot(f,Y)
title('Trasformata di Fourier segnale a tempo continuo')

%% tempo discreto

yd = A*sin(2*pi*f0*n*Tc);
figure
stem(t,yd)
title('Segnale a tempo discreto')

% trasformata di Fourier

N = length(yd);
Yd = 1/N*abs(fftshift(fft(yd)));
F = -Fc/2:Fc/N:Fc/2-Fc/N;
figure
stem(F,Yd)
title('Trasformata di Fourier segnale a tempo discreto')

%% prova DFT

% per Fc-> infinito, la DFT è la Trasformata di Fourier

Y = Tc*fft(y);
f = 0:Fc/N:Fc-Fc/N;
figure, plot(f,abs(Y))
y = ifft(Y)/Tc;
figure, plot(t,y)
