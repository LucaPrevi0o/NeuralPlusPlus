#ifndef CSV_H
#define CSV_H

#include "tensor.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

namespace neural {

    class csv {

    private:

        static constexpr int BUF_SIZE = 4096;
        static constexpr int LINE_SIZE = 1024;
        static constexpr int MAX_COLS = 128;

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
            if (fd < 0) return tensor::tuple<char*>();

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

        static tensor::matrix<float> load(const char* filename, char delimiter = ',') {
            
            int fd = open_file(filename);
            if (fd < 0) return tensor::matrix<float>();

            char buf[BUF_SIZE];
            int bytes_read = 0;
            int buf_pos = 0;
            char line[LINE_SIZE];
            int line_pos = 0;

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
            if (n_rows == 0 || n_cols == 0) return tensor::matrix<float>();

            // Seconda passata: leggi i dati
            fd = open_file(filename);
            if (fd < 0) return tensor::matrix<float>();
            tensor::matrix<float> data(n_rows, n_cols);

            bytes_read = 0;
            buf_pos = 0;
            line_pos = 0;
            int row = 0;
            first_line = true;
            while ((bytes_read = read(fd, buf, BUF_SIZE)) > 0) {

                buf_pos = 0;
                while (buf_pos < bytes_read)
                    if (read_line(buf_pos, bytes_read, buf, line, line_pos)) {

                        if (first_line) first_line = false;
                        else {

                            parse_row(line, delimiter, &data(row, 0), n_cols);
                            row++;
                        }
                    }

            }
            close(fd);
            return data;
        }
    };
}

#endif