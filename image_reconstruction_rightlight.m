files = dir('subject*');
raw_matrix = []; 

% 1. CARICAMENTO PURO
for i = 1 : length(files)
    filename = files(i).name;
    if contains(filename, 'subject')
        I = double(imread(filename));
        raw_matrix = [raw_matrix, I(:)]; 
    end
end

% 2. CENTRATURA GLOBALE (Senza sqrt per semplicità visiva)
mean_face = mean(raw_matrix, 2);
training_centered = raw_matrix - mean_face;

% 3. SVD
[U, ~, ~] = svd(training_centered, 'econ');

% 4. TEST SULL'IMMAGINE
test_img_name = 'subject01.rightlight'; 
I_original = double(imread(test_img_name));

% Centratura test coerente
I_test_centered = I_original(:) - mean_face;

p_values = [1, 5, 15, 50, 100]; 
figure('Name', 'Ricostruzione al variare di p');

subplot(1, length(p_values) + 1, 1);
imshow(I_original, []); title('Original');

for i = 1:length(p_values)
    p = p_values(i);
    Up = U(:, 1:p);
    
    % RICOSTRUZIONE: Proietta, ri-espandi e aggiungi la media
    % Le parentesi (Up' * I_test_centered) evitano l'errore di memoria
    I_rec_vec = Up * (Up' * I_test_centered) + mean_face;
    
    I_rec = reshape(I_rec_vec, size(I_original));
    
    subplot(1, length(p_values) + 1, i + 1);
    imshow(I_rec, []); 
    title(['p = ', num2str(p)]);
end