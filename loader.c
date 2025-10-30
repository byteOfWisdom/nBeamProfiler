#include <stdlib.h>
#include <stdint.h>
#include <strings.h>
#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/signal.h>
#include <math.h>

typedef struct {
  uint64_t time;
  uint8_t channel;  
  double y;
} Event;


typedef struct {
  int64_t filled;
  int64_t len;
  Event* events;
} EventList;


EventList load_file(char*, int64_t, bool);
void extend_event_list(EventList*, uint64_t);
EventList make_event_list(int64_t);
EventList parse_mesy_file(FILE*, uint64_t);


void extend_event_list(EventList* el, uint64_t ext_by){
  Event* old_mem = el->events;
  el->events = (Event*) calloc(el->len + ext_by, sizeof(Event));
  memmove(el->events, old_mem, el->len * sizeof(Event));
  el->len += ext_by;
  free(old_mem);
}

EventList make_event_list(int64_t initial_len) {
  EventList res = {0, initial_len, (Event*) calloc(initial_len, sizeof(Event))};
  return res;
}

Event* test(char* s, int64_t l) {
  EventList all_events;
  all_events.events = (Event*) calloc(l, sizeof(Event));
  FILE* fptr = fopen(s, "r");
  (void) parse_mesy_file(fptr, l);
  fclose(fptr);
  return all_events.events;
}


EventList load_file(char* fname, int64_t load, bool mesy_format) {
  EventList all_events;
  FILE* fptr = fopen(fname, "r");

  if (mesy_format) {
    all_events = parse_mesy_file(fptr, load);
  }
  fclose(fptr);
  // printf("returning now!\n");
  // printf("len now is: %lli\n", all_events.filled);
  // *data_out = all_events->events;
  // printf("wrote len to output\n");
  // return all_events.events;
  return all_events;
}


typedef struct {
  uint64_t timestamp;
  double y[16];
  uint64_t filled;
} MesyLine;

#define buffsize 1024

MesyLine parse_mesy_line(char* line) {
  const double time_const = 1 / 6.25e-8;
  double parse[33];

  for (int i = 0; i < 33; ++i){
    char* chunk = strsep(&line, ",");
    int exists = sscanf(chunk, "%lf", parse + i);
    if (!exists) {
      parse[i] = 0.;
    }
  }

  
  MesyLine res = {0, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}, 0};

  res.timestamp = (uint64_t) round(parse[0] * time_const);

  for (uint8_t i = 0; i < 16; ++i) {
    if ((parse[i + 1] != 0.) & (parse[i + 17] != 0.)) {
      double long_i = parse[i + 1];
      double short_i = parse[i + 17];
      
      res.y[i] = (long_i - short_i) / long_i;
      res.filled ++;
    }
  }
  
  return res;
}


typedef struct {
  pthread_mutex_t reader;
  pthread_mutex_t writer;
  char data[buffsize];
  uint64_t index;
} item_buffer;


item_buffer item_buffer_init(void) {
  item_buffer res = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, "\0", 0};
  pthread_mutex_lock(&res.reader);
  return res;
}


typedef struct {
  item_buffer items[64];
} queue;


typedef struct {
  queue* q_ref;
  MesyLine* line_buffer;
} worker_args;


queue queue_init(void) {
  queue res;
  for (int i = 0; i < 64; ++i) {
    res.items[i] = item_buffer_init();
  }
  return res;
}


item_buffer* find_writable(queue* q) {
  int i = 0;

  while (pthread_mutex_trylock(&q->items[i].writer)) {
    ++ i;
    if (i == 64) i = 0;
  }
  return &q->items[i];
}


item_buffer* find_readable(queue* q) {
  int i = 0;

  while (pthread_mutex_trylock(&q->items[i].reader)){
    ++ i;
    if (i == 64) i = 0;
  }

  return &q->items[i];
}


void wait_queue_finished(queue* q) {
  for (int i = 0; i < 64; ++i) {
    pthread_mutex_lock(&q->items[i].writer);
    // pthread_mutex_lock(&q->items[i].reader);
  }
}

bool worker_run = true;

void* mesy_worker(void* arg) {
  worker_args* args = (worker_args*) arg;
  queue* q = args->q_ref;
  MesyLine* line_buffer = args->line_buffer;

  while (worker_run) {
    item_buffer* item = find_readable(q);
    MesyLine res = parse_mesy_line(item->data);
    res.timestamp = item->index;
    line_buffer[item->index] = res;
    // printf("worker parsed index %llu\n", item->index);
    pthread_mutex_unlock(&item->writer);
  }
  
  return NULL;
}


EventList unpack_events(MesyLine* data_buffer, uint64_t buff_len) {
  printf("enerting unpack!");
  uint64_t total_count = 0;
  for (uint64_t i = 0; i < buff_len; ++i) {
    total_count += data_buffer[i].filled;
  }

  EventList event_list = make_event_list(total_count);
  event_list.filled = total_count;

  printf("there are %llu total events\n", total_count);

  uint64_t n = 0;
  uint64_t i = 0;
  while (n < total_count) {
    for (uint8_t j = 0; j < 16; ++ j) {
      if (data_buffer[i].y[j] != 0.) {
        Event e = {data_buffer[i].timestamp, j, data_buffer[i].y[j]};
        event_list.events[n] = e;
        n ++;
      }
    }
    i += 1;
  }

  printf("finished unpacking!\n");  
  return event_list;
}


EventList parse_mesy_file(FILE* file, uint64_t lines) {
  printf("parsing mesytec file with %llu lines\n", lines);
  const unsigned int thread_count = 7;
  pthread_t worker_handles[thread_count];
  MesyLine* line_data_buffer = (MesyLine*) calloc(lines, sizeof(MesyLine));
  queue data_q = queue_init();

  worker_args args = {&data_q, line_data_buffer};

  for (unsigned int i = 0; i < thread_count; ++i) {
    pthread_create(&worker_handles[i], NULL, mesy_worker, (void*) &args);
  }

  char header[buffsize];
  fgets(header, buffsize, file); 
  lines --;
  (void) header;
  for (uint64_t i = 0; i < lines; ++i){
    item_buffer* item = find_writable(&data_q);
    // printf("reader assgined index %llu\n", i);
    fgets(item->data, buffsize, file);
    item->index = i;
    pthread_mutex_unlock(&item->reader);
  }

  printf("waiting for threads to finish...\n");
  wait_queue_finished(&data_q);

  // printf("threads finished\n");

  printf("unpacking lines\n");

  EventList events = unpack_events(line_data_buffer, lines);
  // EventList events;
  // events.events = (Event*) calloc(lines, sizeof(Event));

  printf("unpacking finished!\n");
  worker_run = false;
  for (unsigned int i = 0; i < thread_count; ++i) {
    printf("setting quit env for thread %i\n", i);
    item_buffer* item = &data_q.items[i];
    item->index = 0;
    strcpy(item->data, "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n\0");
    pthread_mutex_unlock(&item->reader);
  }

  for (unsigned int i = 0; i < thread_count; ++i) {
    pthread_join(worker_handles[i], NULL);
  }
  free(line_data_buffer);
  printf("c has finished, there are %llu\n", events.filled);

  // for (uint64_t i = 0; i < events.filled; i += 100) {
  //   printf("sample read: %llu\n", events.events[i].time);
  // }
  return events;
}
