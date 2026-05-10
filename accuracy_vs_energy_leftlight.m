files = dir('subject*'); % Legge tutti i file nella cartella corrente
training_matrix = []; % Matrice che conterrà le facce (colonne)
labels = {}; % Per ricordarci a quale file corrisponde ogni colonna
test_expression = 'leftlight';    % L'espressione usata per il test

% Ciclo FOR per costruire il Database
for i = 1 : length(files)
    filename = files(i).name;
    
    % Prendiamo solo i file che contengono 'subject' 
    % ma escludiamo quelli che contengono la nostra test_expression
    if contains(filename, 'subject') && ~contains(filename, test_expression)
        
        % Leggiamo l'immagine
        I = imread(filename);
        
        % Pre-processing
        I_vec = double(I(:));   % Trasforma la matrice in un vettore colonna
        
        % Aggiungiamo alla matrice del database
        training_matrix = [training_matrix, I_vec]; 
        
        % Salviamo il nome per sapere chi è chi
        labels = [labels, filename];
    end
end

%We subtract the mean face to center the dataset
mean_face = mean(training_matrix, 2);
n = size(training_matrix, 2);
training_matrix = 1/sqrt(n) * (training_matrix - mean_face); 

% Parametri dell'esperimento
valori_p = [1, 2, 3, 5, 10, 20, 30, 50, 100, 150]; % I valori di p che vogliamo testare
accuratezza = zeros(size(valori_p));   % Vettore per salvare i risultati
energia_p = zeros(size(valori_p));      

% Calcolo SVD
[U, S, ~] = svd(training_matrix, 'econ');
s = diag(S); % Valori singolari
energia_totale = sum(s.^2);
energia_cumulata = cumsum(s.^2) / energia_totale;

% 2. Ciclo sui diversi valori di p
for k = 1:length(valori_p)
    p = valori_p(k);
    energia_p(k) = energia_cumulata(p) * 100;
    Up = U(:, 1:p);
    F_train = Up' * training_matrix;
    
    successi = 0;
    totale_test = 0;
    
    % 3. Ciclo di test su tutte le immagini escluse
    for i = 1:length(files)
        filename = files(i).name;
        if contains(filename, 'subject') && contains(filename, test_expression)
            totale_test = totale_test + 1;
            
            % Preprocessing immagine di test
            I_test = double(imread(filename));
            I_test = 1/sqrt(n) * (I_test(:) - mean_face);
            
            % Proiezione e ricerca del vicino
            f_test = Up' * I_test;
            distanze = sqrt(sum((F_train - f_test).^2, 1));
            [~, idx] = min(distanze);
            
            % Verifica (confronto i primi 9 caratteri 'subjectXX')
            if strncmp(filename, labels{idx}, 9)
                successi = successi + 1;
            else
                fprintf('Soggetto %s fallito: scambiato per %s\n', filename, labels{idx})
            end
        end
    end
    
    accuratezza(k) = (successi / totale_test) * 100;
    fprintf('p = %d | Accuratezza: %.2f%%\n', p, accuratezza(k));

end

% 4. Visualizzazione dei risultati
figure;
plot(valori_p, accuratezza, '-o', 'LineWidth', 2);
hold on
plot(valori_p, energia_p, '-s', 'LineWidth', 2);
ylim([0 100]);
xlabel('Number of singular values (p)');
title('Accuracy vs Energy');
legend('Accuracy', 'Energy');
grid on;