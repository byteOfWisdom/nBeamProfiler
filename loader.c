#include <stdlib.h>
#include <stdint.h>
#include <strings.h>
#include <stdio.h>
#include <stdbool.h>

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
Event* load_mesy_file(char*);
void extend_event_list(EventList*, uint64_t);
EventList make_event_list(uint64_t);


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
  (void) fname; 
  (void) mesy_format;
  EventList all_events = make_event_list(load);
  for (uint64_t i = 0; i < load; ++i) {
    all_events.events[i] = (Event) {i, 0.0, 0};
  }
  all_events.filled = load;
  return all_events;
}

