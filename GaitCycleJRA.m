N=po
%for Ankle Angle
b = smooth(AnkleAngle(:));
B1 = reshape(b,100,1);
%Plot the original data and the smoothed data:

subplot(2,1,2)
plot(AnkleAngle,':');
hold on
plot(B1,'-','LineWidth',2);

title('Smooth B1 for Ankle(All Data)')
xlabel('Location')
ylabel('Ankle Angle Degree')

[pks,locs] = findpeaks(B1);
[peaks,loc,width,prom] = findpeaks(B1,1:size(B1,1),'MinPeakProminence',10, 'MinPeakWidth', 7.4, 'MaxPeakWidth', 30)

[pks1,locs1] = findpeaks(AnkleAngle);
[peaks1,loc1,width1,prom1] = findpeaks(AnkleAngle,1:size(AnkleAngle,1),'MinPeakProminence',10, 'MinPeakWidth', 7.4, 'MaxPeakWidth', 30)
%-------Added parts from patrick ----------------------------------------------------------------------------------------------

FrameStart = loc(1);        % If valid, the start of the gait cycle begins at the location of peak 1
FrameEnd = loc(2);          % If valid, the end of the gait cycle happens at the location of peak 2


line([FrameStart FrameStart], get(gca, 'YLim'),'Color',[.5 .5 .5],'LineStyle','--','LineWidth',2);
line([FrameEnd FrameEnd], get(gca, 'YLim'),'Color',[.5 .5 .5],'LineStyle','--','LineWidth',2);
% For Knee Angles
c = smooth(KneeAngles(:));
C1 = reshape(c,100,2);
%Plot the original data and the smoothed data:

subplot(2,1,1)
plot(KneeAngles,':');
hold on
plot(C1,'-','LineWidth',2);

title('Smooth C1 for knee(All Data)')
xlabel('Location')
ylabel('Knee Angle Degree')

line([FrameStart FrameStart], [0 200],'Color',[.5 .5 .5],'LineStyle','--','LineWidth',2);   
line([FrameEnd FrameEnd], [0 200],'Color',[.5 .5 .5],'LineStyle','--','LineWidth',2);

KneeAngleL = KneeAngles(:,1);            %Reading the JRA data for Left Knee
KneeAngleR = KneeAngles(:,2);            %Reading the JRA data for Right Knee

%%------------------------------ to find min Left and Right Knee Angles
AngleDataR = smooth(KneeAngleR);
AngleDataL = smooth(KneeAngleL);


angledegR_T = AngleDataR(FrameStart:FrameEnd);
angledegL_T = AngleDataL(FrameStart:FrameEnd);

% Finding lowest point in the gait cycle
MinangledegR = min(angledegR_T);
MinangledegL = min(angledegL_T);




