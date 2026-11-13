import type { BackendProject, BackendSurvey, BackendWell } from '../../api/backend';
import type { DataItem, SeismicData, TableData, WellData } from '../data';

/** Shared application data and actions required by tab views. */
export type TabContext = Record<string, any> & {
  active?: DataItem;
  activeSeismic?: SeismicData;
  activeTable?: TableData;
  activeWell?: WellData;
  items: DataItem[];
  projects: BackendProject[];
  selectedProject: BackendProject | null;
  selectedSeismicDataset?: DataItem;
  surveySeismicItems: DataItem[];
  surveys: BackendSurvey[];
  wells: BackendWell[];
};
