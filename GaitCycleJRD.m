%Gait Cycle DEMO
%By Patrick Kozlow

clear;
close all;

%Initializing Variables
subjects = 1;       %Number of subjects data is collected for
sequences = 1;      %Number of sequences per subject
       
for sub_num = 1:subjects            %Outer loop for number of people
    for run_num = 1:sequences       %Inner loop for number of runs per person
       
subsec1 = ['Subject_' num2str(sub_num) '_Seq_' num2str(run_num) '_GaitCycle_N.csv'];    %loading the JRD data
sequence = readtable(subsec1);          %Reading the JRD data
GaitCycleData = sequence{:,1};          %Reading the JRD data  
GaitCycleData = smooth(GaitCycleData);  %Smoothing the data    

% plotting the filtered graph
hFig = figure(1);                           %Creating a figure
set(hFig, 'Position', [600,200,600,600]);   %Defining position of figure on screen
subplot(2,1,1);                             %Initializing 2 plots on the figure
plot(GaitCycleData,'LineWidth',2);          %Plotting smoothed JRD data
axis([0 100 0 0.8]);                        %Creating the axis    
ylabel('\Delta between ankles');            %Creating Y axis label
xlabel('Frames');                           %Creating X axis label

title(['Person ' num2str(sub_num) ' Sequence ' num2str(run_num)]);  %Creating the title for the graph

FrameStart = 1;                     %Initializing variables for the next part (first frame)
FrameEnd = size(GaitCycleData,1);   %Initializing variables for the next part (last frame)

% Finding the peaks to be used for finding gait cycle 
% In this part, we find and store all the valid peaks
% A 'valid' peak is a peak that has:
% A minimum prominence of:  0.13
% A minimum width of:       7.4 frames
% A maximum width of:       18 frames
[peaks,loc,width,prom] = findpeaks(GaitCycleData(10:FrameEnd),10:size(GaitCycleData,1),'MinPeakProminence',0.13, 'MinPeakWidth', 7.4, 'MaxPeakWidth', 18);

% Checking to see if gait cycle is valid
% For the JRD method to work, we need to find 3 valid peaks
% This 'if' statement checks to see if there are 3 valid peaks
if size(loc) < 3
   disp(['Invalid Data for person ' num2str(sub_num) ' sequence ' num2str(run_num)]);
   GaitCycleSpeed(sub_num,run_num) = 0;
   
else   
FrameStart = loc(1);        % If valid, the start of the gait cycle begins at the location of peak 1
FrameEnd = loc(3);          % If valid, the end of the gait cycle happens at the location of peak 3

% Adding lines to represent where gait cycle starts and stops 
line([FrameStart FrameStart], get(gca, 'YLim'),'Color',[.5 .5 .5],'LineStyle','--','LineWidth',2);
line([FrameEnd FrameEnd], get(gca, 'YLim'),'Color',[.5 .5 .5],'LineStyle','--','LineWidth',2);
end
%Insert a breakpoint here to test if the first plot generates correctly

%% KneeAngles
%Here we will plot the knee angles and overlay the gait cycle found earlier

subsec2 = ['Subject_' num2str(sub_num) '_Seq_' num2str(run_num) '_JointAngles_N.csv'];  %loading the JRA data
sequence = readtable(subsec2);          %Reading the JRA data
AngleData = sequence{:,[1,2]};          %Reading the JRA data

AngleDataL = AngleData(:,1);            %Reading the JRA data for Left Knee
AngleDataR = AngleData(:,2);            %Reading the JRA data for Right Knee

AngleDataL = smooth(AngleDataL);        %Smoothing the JRA data for Left Knee
AngleDataR = smooth(AngleDataR);        %Smoothing the JRA data for Right Knee

subplot(2,1,2);                         %Selecting second plot in the figure
plot(AngleDataL,'LineWidth',2);         %Plotting the Left knee angle
hold on
plot(AngleDataR,'LineWidth',2);         %Plotting the Right knee angle

% Adding lines to represent where gait cycle starts and stops on the Angle graph
line([FrameStart FrameStart], [0 200],'Color',[.5 .5 .5],'LineStyle','--','LineWidth',2);   
line([FrameEnd FrameEnd], [0 200],'Color',[.5 .5 .5],'LineStyle','--','LineWidth',2);

legend('Left Knee Angle','Right Knee Angle','Location','southwest');    %Adding a legend
axis([0 100 0 200]);                %Creating the axis
ylabel('Angle in Degrees');         %Labeling Y axis
xlabel('Frames');                   %Labeling X axis
    end
end
%%------------------------------ to find min Left and Right Knee Angles
%angledegR = transpose(angledegR);

AngleDataR = smooth(AngleDataR);
AngleDataL = smooth(AngleDataL);


angledegR_T = AngleDataR(FrameStart:FrameEnd);
angledegL_T = AngleDataL(FrameStart:FrameEnd);

% Finding lowest point in the gait cycle
MinangledegR = min(angledegR_T);
MinangledegL = min(angledegL_T);
%%-------------------------------
%angledegR = transpose(angledegR);
%angledegR = smooth(angledegR);

%angledegR_T = angledegR(FrameStart:FrameEnd);
%angledegL_T = angledegL(FrameStart:FrameEnd);

% Finding lowest point in the gait cycle
%MinangledegR = min(angledegR_T);
%MinangledegL = min(angledegL_T);
