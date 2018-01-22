data = load('data.txt');

% Define x and y
x = data(:,1);
y = data(:,2);

% Create a function to plot the data
function plotData(x,y)
  plot(x,y,'rx','MarkerSize',8); % Plot the data
end

% Plot the data
plotData(x,y);
xlabel('weight'); % Set the x-axis label
ylabel('length'); % Set the y-axis label

% Count how many data points we have
m = length(x);
% Add a column of all ones (intercept term) to x
X = [ones(m, 1) x];

theta = (pinv(X'*X))*X'*y

% Plot the fitted equation we got from the regression
hold on; % this keeps our previous plot of the training data visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % Don't put any more plots on this figure