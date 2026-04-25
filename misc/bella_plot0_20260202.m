%% Making the first overview plot

%% Move to the data directory
cd('/mnt/data/bjorn/EISCAT/Analysed/2026-02-02_bella_30@vhf/')
%% Read in the results
% The results from the GUISDAP analysis are saved in individual
% mat-files. The guisdap_param2cell2regular reads those files and returns
% the results in a sensible format
q = dir('./*.mat');
[h,t,ne,Te,Ti,vi,dne,dTe,dTi,dvi,az,el,T] = guisdap_param2cell2regular(q,...
                                          [2026 2 2 18 0 0;2026 2 2 23 20 0]);
%% Make a nice plot.
% Here one can spend much time finessing the esthetics of the presentation
figure
colormap(turbo)
subplot(4,1,1)
pcolor(rem(t/3600,24),h,log10(max(1e8,ne))),shading flat
caxis([9 12.5])
timetick
set(gca,'TickDir','both','box','off','fontsize',11)
cbh1 = colorbar_labeled('m^{-3}','log','fontsize',12);
ylabel('alt (km)')
ylabel('')
title('Electron density')
subplot(4,1,2)
pcolor(rem(t/3600,24),h,Te),shading flat                
caxis([0 4500])
timetick
set(gca,'TickDir','both','box','off','fontsize',11)
ylabel('altitude (km)')
title('Electron Temperature')
cbh2 = colorbar_labeled('K','linear','fontsize',12);
subplot(4,1,3)
pcolor(rem(t/3600,24),h,Ti),shading flat                
caxis([0 2500])
timetick
set(gca,'TickDir','both','box','off','fontsize',11)
title('Ion Temperature')
cbh3 = colorbar_labeled('K','linear','fontsize',12);
subplot(4,1,4)
pcolor(rem(t/3600,24),h,vi),shading flat                
caxis([-500 500])
timetick
xlabel('Time (UT)')
set(gca,'TickDir','both','box','off','fontsize',11)
set(gca,'colormap',redblue)
title('Ion l-o-s velocity')
cbh_vi = colorbar_labeled('m/s','linear','fontsize',12);
set(cbh_vi,'colormap',redblue)
set(cbh1,'Position',get(cbh1,'Position')+[-0.0125 0 0 0])
set(cbh2,'Position',get(cbh2,'Position')+[-0.0125 0 0 0])
set(cbh3,'Position',get(cbh3,'Position')+[-0.0125 0 0 0])
set(cbh_vi,'Position',get(cbh_vi,'Position')+[-0.0125 0 0 0])
%% Here the data are saved into one mat-file for convenient use
save('bella_20260202.mat',...
     'h','t','ne','Te','Ti','vi',...
     'dne','dTe','dTi','dvi','az','el','T')