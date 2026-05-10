% 1. Parametri dell'esperimento
soggetto_da_escludere = 'subject03';
p_fisso = 20;

% 2. Caricamento e separazione Training (Tutti tranne subject03) e Test (Solo subject03)
files = dir('subject*');
training_matrix = [];
labels_train = {};
test_images = [];
test_labels = {};

for i = 1:length(files)
    filename = files(i).name;
    I = double(imread(filename));
    
    if contains(filename, soggetto_da_escludere)
        test_images = [test_images, I(:)];
        test_labels{end+1} = filename;
    else
        training_matrix = [training_matrix, I(:)];
        labels_train{end+1} = filename;
    end
end

% 3. Training (SVD sul database senza subject03)
mean_face = mean(training_matrix, 2);
n_train = size(training_matrix, 2);
A_train = (training_matrix - mean_face) / sqrt(n_train);
[U, ~, ~] = svd(A_train, 'econ');
Up = U(:, 1:p_fisso);
F_train = Up' * A_train;

% 4. Test e Raccolta Distanze
distanze_minime = zeros(1, size(test_images, 2));

fprintf('--- Test su Soggetto Sconosciuto (%s) ---\n', soggetto_da_escludere);
for j = 1:size(test_images, 2)
    % Proiezione coerente (usando n_train)
    f_test = Up' * ((test_images(:, j) - mean_face) / sqrt(n_train));
    
    % Calcolo distanze euclidee
    diff = F_train - f_test;
    distanze = sqrt(sum(diff.^2, 1));
    [dist_min, idx_min] = min(distanze);
    
    distanze_minime(j) = dist_min;
    fprintf('Test: %s -> Match: %s | Distanza: %.2f\n', ...
        test_labels{j}, labels_train{idx_min}, dist_min);
end

mean(distanze_minime);
fprintf('Media distanze minime: %s', mean(distanze_minime))

% 5. Visualizzazione dei Risultati
figure;
stem(distanze_minime, 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
hold on;

% Estetica del grafico
xticks(1:length(test_labels));
xticklabels(strrep(test_labels, '_', '\_')); % Sostituisce _ con \_ per visualizzare correttamente i nomi
xtickangle(45);
ylabel('Minimum Euclidean Distance');
title(['Unknown Subject Rejection Analysis (Excluded: ', soggetto_da_escludere, ')']);
grid on;

% Aggiunta di una linea di soglia ipotetica (es. a 400 se i tuoi dati sono intorno a 500)
% Nota: regola il valore 400 in base ai tuoi risultati effettivi
soglia = 350; 
yline(soglia, 'r--', 'Threshold \tau', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 2);

legend('Test Image Distance', 'Rejection Threshold');