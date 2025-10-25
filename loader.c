#include <stdlib.h>
#include <stdint.h>
#include <strings.h>
#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/_pthread/_pthread_cond_t.h>
#include <sys/_pthread/_pthread_mutex_t.h>

typedef struct {
  uint64_t time;
  double y;
  uint8_t channel;  
} Event;


typedef struct {
  uint64_t filled;
  uint64_t len;
  Event* events;
} EventList;


EventList load_file(char*, int64_t, bool);
void extend_event_list(EventList*, uint64_t);
EventList make_event_list(uint64_t);
void parse_mesy_file(FILE*, EventList*, uint64_t);


void extend_event_list(EventList* el, uint64_t ext_by){
  Event* old_mem = el->events;
  el->events = (Event*) calloc(el->len + ext_by, sizeof(Event));
  memmove(el->events, old_mem, el->len * sizeof(Event));
  el->len += ext_by;
  free(old_mem);
}

EventList make_event_list(uint64_t initial_len) {
  EventList res = {0, initial_len, (Event*) calloc(initial_len, sizeof(Event))};
  return res;
}


EventList load_file(char* fname, int64_t load, bool mesy_format) {
  EventList all_events = make_event_list(load);
  FILE* fptr = fopen(fname, "r");

  if (mesy_format) {
    parse_mesy_file(fptr, &all_events, load);
  }
  return all_events;
}


typedef struct {
  uint64_t timestamp;
  double y[16];
} MesyLine;

#define buffsize 1024

typedef struct {
  pthread_cond_t process;
  pthread_mutex_t processing;
  bool should_terminate;
  char buffer[buffsize];
  uint64_t index;
  uint64_t event_count;
  MesyLine* line_list;
} WorkerInterface;


MesyLine parse_mesy_line(char* line) {
  MesyLine res = {0, {1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}};
  // printf("parsed line \n");
  (void) line;
  return res;
}


void* mesy_worker(void* arg) {
  // printf("worker started\n");
  WorkerInterface* interface = (WorkerInterface*) arg;
  MesyLine* line_list = interface->line_list;

  printf("worker entering loop \n");
  while (!interface -> should_terminate){
    // printf("waiting for parse\n");

    pthread_cond_wait(&interface->process, &interface->processing);
    // printf("got condition\n");
    // pthread_mutex_lock(&interface->processing);
    
    // printf("parsing line\n");
    MesyLine parsed_line = parse_mesy_line(interface -> buffer);
    line_list[interface->index] = parsed_line;
    for (int i = 0; i < 16; ++ i){
      if (parsed_line.y[i] != 0) ++interface->event_count;
    }

    // printf("finished parse \n");

    // pthread_mutex_unlock(&interface->processing);
  }

  // printf("exiting thread! \n");
  return NULL;
}


void parse_mesy_file(FILE* file, EventList* events, uint64_t lines) {
  printf("parsing mesytec file with %llu lines\n", lines);
  const unsigned int thread_count = 8;
  WorkerInterface workers[thread_count];
  pthread_t worker_handles[thread_count];
  MesyLine* line_data_buffer = (MesyLine*) calloc(lines, sizeof(MesyLine));

  for (unsigned int i = 0; i < thread_count; ++i) {
    workers[i] = (WorkerInterface) {{}, {}, false, {"\0"}, 0, 0, line_data_buffer};
    pthread_cond_init(&workers[i].process, NULL);
    pthread_mutex_init(&workers[i].processing, NULL);
    pthread_create(&worker_handles[i], NULL, mesy_worker, (void*) &workers[i]);
  }

  unsigned char worker = 0; // resets via overflow

  for (uint64_t i = 0; i < lines; ++i){
    while (pthread_mutex_trylock(&workers[worker].processing)){
      ++ worker;
      if (worker == thread_count) worker = 0;
    }
    fgets(workers[worker].buffer, buffsize, file);
    workers[worker].index = i;
    pthread_mutex_unlock(&workers[worker].processing);
    pthread_cond_signal(&workers[worker].process);
    // printf("assigned line %llu to worker %i\n", i, worker);
  }

  printf("stopping workers\n");
  // unpack parsed lines
  uint64_t total_event_count = 0;
  for (unsigned int i = 0; i < thread_count; ++i) {
    total_event_count += workers[i].event_count;
    pthread_mutex_lock(&workers[i].processing);
    workers[i].should_terminate = true;
    pthread_cond_signal(&workers[i].process);
  }

  printf("unpacking lines\n");
  if (total_event_count > events->len)
    extend_event_list(events, total_event_count - events->len);

  events->filled = total_event_count;
  printf("%llu\n", total_event_count);
  for (uint64_t i = 0; i < lines; ++i) {
    uint64_t n = 0;
    for (uint8_t j = 0; j < thread_count; ++j) {
      if (line_data_buffer[i].y[j] != 0){
        events->events[i + n] = (Event) {line_data_buffer[i].timestamp, line_data_buffer[i].y[j], j};
      }
    }
  }

  free(line_data_buffer);
}
