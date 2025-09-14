#ifndef CSV_H
#define CSV_H

#include "tensor/tensor.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

namespace neural {

    class csv {

    private:

        static constexpr int BUF_SIZE = 4096;
        static constexpr int LINE_SIZE = 1024;
        static constexpr int MAX_COLS = 128;

        struct data {

            tensor::matrix<float> features;
            tensor::matrix<float> targets;

            data(int n_features, int n_targets, int n_samples) : 
                features(n_features, n_samples), targets(n_targets, n_samples) {}

            data(const tensor::matrix<float>& f, const tensor::matrix<float>& t) : features(f), targets(t) {}
        };

        // Funzione per aprire un file in sola lettura
        static int open_file(const char* filename) { return open(filename, O_RDONLY); }

        // Funzione per leggere una riga dal file
        static bool read_line(int& buf_pos, int bytes_read, char* buf, char* line, int& line_pos) {

            while (buf_pos < bytes_read) {

                char c = buf[buf_pos++];
                if (c == '\n' || c == '\r') {
                    if (line_pos > 0) {

                        line[line_pos] = '\0';
                        line_pos = 0;
                        return true;
                    }
                } else if (line_pos < LINE_SIZE - 1) line[line_pos++] = c;
            }
            return false;
        }

        // Funzione per contare le colonne in una riga
        static int count_columns(const char* line, char delimiter) {

            int n_cols = 1;
            for (int i = 0; line[i] != '\0'; ++i) if (line[i] == delimiter) n_cols++;
            return n_cols;
        }

        // Parsing di una riga CSV in una tupla di char*
        static tensor::tuple<char*> parse_labels(const char* line, char delimiter) {

            char* tokens[MAX_COLS];
            int n_labels = 0;
            char* token = (char*)line;
            char* end = (char*)line;
            while (*end) {

                if (*end == delimiter) {

                    *end = '\0';
                    int len = end - token;
                    char* label = new char[len + 1];
                    for (int i = 0; i < len; ++i) label[i] = token[i];
                    label[len] = '\0';
                    tokens[n_labels++] = label;
                    token = end + 1;
                }
                end++;
            }
            if (token[0] != '\0') {

                int len = end - token;
                char* label = new char[len + 1];
                for (int i = 0; i < len; ++i) label[i] = token[i];
                label[len] = '\0';
                tokens[n_labels++] = label;
            }

            tensor::tuple<char*> labels(n_labels);
            for (int i = 0; i < n_labels; ++i) labels(i) = tokens[i];
            return labels;
        }

        // Parsing di una riga CSV in una tupla di float
        static void parse_row(const char* line, char delimiter, float* out, int n_cols) {

            int col = 0;
            char* token = (char*)line;
            char* end = (char*)line;
            while (*end && col < n_cols) {
                
                if (*end == delimiter) {

                    *end = '\0';
                    out[col++] = atof(token);
                    token = end + 1;
                }
                end++;
            }
            if (token[0] != '\0' && col < n_cols) out[col] = atof(token);
        }

        static tensor::tuple<char*> get_labels(const char* filename, char delimiter = ',') {
            
            int fd = open_file(filename);
            if (fd < 0) return tensor::tuple<char*>(0);

            char buf[BUF_SIZE];
            int bytes_read = 0;
            int buf_pos = 0;
            char line[LINE_SIZE];
            int line_pos = 0;

            tensor::tuple<char*> labels;
            bool found = false;
            while (!found && (bytes_read = read(fd, buf, BUF_SIZE)) > 0) {

                buf_pos = 0;
                while (!found && buf_pos < bytes_read) if (read_line(buf_pos, bytes_read, buf, line, line_pos)) {
                
                    labels = parse_labels(line, delimiter);
                    found = true;
                }
            }
            close(fd);
            return labels;
        }

    public:

        static tensor::matrix<float> load(const char* filename, char delimiter = ',', int max_rows = 0) {
            
            int fd = open_file(filename);
            if (fd < 0) return tensor::matrix<float>(0, 0);

            char buf[BUF_SIZE];
            int bytes_read = 0;
            int buf_pos = 0;
            char line[LINE_SIZE];
            int line_pos = 0;

            int num_rows_read = 0;

            // Prima passata: conta righe e colonne
            int n_rows = 0, n_cols = 0;
            bool first_line = true;
            while ((bytes_read = read(fd, buf, BUF_SIZE)) > 0) {

                buf_pos = 0;
                while (buf_pos < bytes_read) if (read_line(buf_pos, bytes_read, buf, line, line_pos)) {
                    if (first_line) {

                        n_cols = count_columns(line, delimiter);
                        first_line = false;
                    } else n_rows++;
                }
            }
            close(fd);

            // Se non ci sono dati
            if (n_rows == 0 || n_cols == 0) return tensor::matrix<float>(0, 0);

            if (max_rows && n_rows > max_rows) n_rows = max_rows;

            // Seconda passata: leggi i dati
            fd = open_file(filename);
            if (fd < 0) return tensor::matrix<float>(0, 0);

            // Create transposed matrix: features x samples (network format)
            tensor::matrix<float> data(n_cols, n_rows);

            bytes_read = 0;
            buf_pos = 0;
            line_pos = 0;
            int row = 0;
            first_line = true;
            
            // Temporary buffer for parsing a row
            float* temp_row = new float[n_cols];

            while ((bytes_read = read(fd, buf, BUF_SIZE)) > 0) {

                buf_pos = 0;
                while ((max_rows && num_rows_read <= max_rows) && buf_pos < bytes_read) if (read_line(buf_pos, bytes_read, buf, line, line_pos)) {

                    if (first_line) first_line = false;
                    else {

                        parse_row(line, delimiter, temp_row, n_cols);
                        // Transpose while storing: data[feature][sample] = temp_row[feature]
                        for (int col = 0; col < n_cols; col++) data(col, row) = temp_row[col];
                        row++;
                    }
                num_rows_read++;
                }
            }
            close(fd);
            delete[] temp_row;
            return data;
        }

        // New method to load data and split into features and targets
        static data load_split(const char* filename, int target_cols = 1, int max_rows = 0, char delimiter = ',') {

            auto full_data = load(filename, delimiter, max_rows);
            if (full_data.size(0) == 0 || full_data.size(1) == 0)return data(0, 0, 0);

            int n_features = full_data.size(0) - target_cols;
            int n_samples = full_data.size(1);

            if (n_features <= 0) throw "Not enough columns for features";

            // Extract features (first n_features rows)
            auto features = tensor::matrix<float>(n_features, n_samples);
            for (int i = 0; i < n_features; i++)
                for (int j = 0; j < n_samples; j++) features(i, j) = full_data(i, j);

            // Extract targets (last target_cols rows)
            auto targets = tensor::matrix<float>(target_cols, n_samples);
            for (int i = 0; i < target_cols; i++)
                for (int j = 0; j < n_samples; j++) targets(i, j) = full_data(n_features + i, j);

            return data(features, targets);
        }
    };
}

#endif