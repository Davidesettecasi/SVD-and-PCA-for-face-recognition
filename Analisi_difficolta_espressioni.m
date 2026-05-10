% Lista delle espressioni tipiche del dataset Yale
espressioni = {'centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', ...
               'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink'};

risultati_tipologia = zeros(length(espressioni), 1);
p_fisso = 20; % Puoi anche mettere questo dentro un ciclo per p


% --- Caricamento iniziale (una sola volta) ---
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

% --- Ciclo sulle Tipologie ---
for e = 1:length(espressioni)
    tipo_test = espressioni{e};
    fprintf('Testando esclusione di: %s... ', tipo_test);
    
    % Indici delle immagini che appartengono alla tipologia di test
    idx_test = contains(full_labels, tipo_test);
    idx_train = ~idx_test;
    
    % Separazione Training e Test
    training_set = full_matrix(:, idx_train);
    training_labels = full_labels(idx_train);
    test_set = full_matrix(:, idx_test);
    test_labels = full_labels(idx_test);
    
    % Pre-processing (Media Globale del Training)
    mean_face = mean(training_set, 2);
    A_train = (training_set - mean_face) / sqrt(size(training_set, 2));
    
    % SVD e Proiezione Training
    [U, S, ~] = svd(A_train, 'econ');
    Up = U(:, 1:p_fisso);
    F_train = Up' * A_train;
    
    % Test su tutte le immagini della tipologia esclusa
    successi = 0;
    for j = 1:size(test_set, 2)
        f_test = Up' * 1/sqrt(size(training_set, 2)) * (test_set(:, j) - mean_face);
        distanze = sqrt(sum((F_train - f_test).^2, 1));
        [~, idx_min] = min(distanze);
        
        if strncmp(test_labels{j}, training_labels{idx_min}, 9)
            successi = successi + 1;
        end
    end
    
    risultati_tipologia(e) = (successi / size(test_set, 2)) * 100;
    fprintf('Accuratezza: %.2f%%\n', risultati_tipologia(e));
end


figure;
bar(risultati_tipologia);
set(gca, 'XTickLabel', espressioni, 'XTickLabelRotation', 45);
ylabel('Accuracy (%)');
title(['Recognition Accuracy across Different Missing Facial Conditions (p = ', num2str(p_fisso), ')']);
grid on;