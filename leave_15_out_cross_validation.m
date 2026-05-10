% --- Parametri ---
k = 15;               % Numero di immagini da escludere (Test Set)
num_iter = 200;        % Numero di ripetizioni per la statistica
p_fisso = 20;         % Numero di Eigenfaces
acc_random = zeros(num_iter, 1);

% Caricamento (assumendo di avere tutto nella stessa cartella)
files = dir('subject*');
full_matrix = [];
full_labels = {};

for i = 1:length(files)
    filename = files(i).name;
    
    % Proviamo a leggere l'immagine. Usiamo try-catch per saltare file corrotti
    try
        % Specifichiamo il formato se necessario, o leggiamo semplicemente
        I = imread(filename); 
        
        % Se l'immagine è 2D, la vettorizziamo
        full_matrix = [full_matrix, double(I(:))];
        full_labels{end+1} = filename; % Usiamo le graffe per le cell array
    catch
        fprintf('Salto il file: %s (non sembra un''immagine valida)\n', filename);
    end
end

num_totale = size(full_matrix, 2);

fprintf('Avvio Leave-%d-out (%d iterazioni)...\n', k, num_iter);

for it = 1:num_iter
    % 1. Selezione casuale degli indici
    indici_casuali = randperm(num_totale);
    idx_test = indici_casuali(1:k);
    idx_train = indici_casuali(k+1:end);
    
    % 2. Separazione Training e Test
    train_set = full_matrix(:, idx_train);
    train_labels = full_labels(idx_train);
    test_set = full_matrix(:, idx_test);
    test_labels = full_labels(idx_test);
    
    % 3. Training
    mean_face = mean(train_set, 2);
    A_train = (train_set - mean_face) / sqrt(size(train_set, 2));
    [U, ~, ~] = svd(A_train, 'econ');
    Up = U(:, 3:p_fisso);
    F_train = Up' * A_train;
    
    % 4. Test
    successi = 0;
    for j = 1:k
        f_test = Up' * (test_set(:, j) - mean_face) / sqrt(size(train_set, 2));
        distanze = sqrt(sum((F_train - f_test).^2, 1));
        [~, idx_min] = min(distanze);
        
        if strncmp(test_labels{j}, train_labels{idx_min}, 9)
            successi = successi + 1;
        end
    end
    
    acc_random(it) = (successi / k) * 100;
end

% --- Risultati ---
fprintf('\n--- RISULTATI FINALI ---\n');
fprintf('Accuratezza Media: %.2f%%\n', mean(acc_random));
fprintf('Deviazione Standard: %.2f%%\n', std(acc_random));

% Istogramma per il report
figure;
histogram(acc_random, 10, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Accuratezza (%)');
ylabel('Frequenza (N. Iterazioni)');
title(['Distribution of Accuracy for Leave-', num2str(k), '-out CV']);
grid on;