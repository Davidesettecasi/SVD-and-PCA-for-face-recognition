files = dir('subject*'); 
training_matrix = []; 
labels = {}; 
test_expression = 'sleepy'; 

% --- 1. COSTRUZIONE DATABASE (TRAINING) ---
for i = 1 : length(files)
    filename = files(i).name;
    if ~contains(filename, test_expression)
        I = double(imread(filename));
        training_matrix = [training_matrix, I(:)]; 
        labels = [labels, filename];
    end
end

mean_face = mean(training_matrix, 2);
n_train = size(training_matrix, 2);
% Normalizzazione coerente
training_matrix_norm = (training_matrix - mean_face) / sqrt(n_train); 

[U, ~, ~] = svd(training_matrix_norm, 'econ');
p = 20;
Up = U(:, 1:p);
F_train = Up' * training_matrix_norm;

% --- 2. TEST SU TUTTI I SOGGETTI NOTI (VERSIONE SLEEPY) ---
distanze_minime_noti = [];

fprintf('--- Analisi Distanze Soggetti Noti (Espressione: %s) ---\n', test_expression);

for i = 1 : length(files)
    filename = files(i).name;
    
    if contains(filename, test_expression)
        I_test = double(imread(filename));
        % Centratura e normalizzazione identica al training
        I_test_centered = (I_test(:) - mean_face) / sqrt(n_train);
        f_test = Up' * I_test_centered;
        
        % Calcolo distanze
        distanze = sqrt(sum((F_train - f_test).^2, 1));
        [valore_minimo, ~] = min(distanze);
        
        distanze_minime_noti = [distanze_minime_noti, valore_minimo];
    end
end

% --- 3. CALCOLO STATISTICHE ---
media_noti = mean(distanze_minime_noti);
dev_std_noti = std(distanze_minime_noti);
max_noti = max(distanze_minime_noti);

fprintf('\nStatistiche Distanze Soggetti Noti:\n');
fprintf('Media: %.4f\n', media_noti);
fprintf('Deviazione Standard: %.4f\n', dev_std_noti);
fprintf('Distanza Massima: %.4f\n', max_noti);

% Visualizzazione per il report
figure;
histogram(distanze_minime_noti, 8, 'FaceColor', [0.4 0.8 0.4]);
xlabel('Euclidean Distance');
ylabel('Frequency');
title(['Distribution of Distances for Known Subjects (p=', num2str(p), ')']);
grid on;